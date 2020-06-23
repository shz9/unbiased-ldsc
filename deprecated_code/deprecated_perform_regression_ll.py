import pandas as pd
import numpy as np
import os
import errno
import sys
import glob
import pickle
import matplotlib.pylab as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from multiprocessing import Pool
from ldsc.ldscore.regressions import Hsq


def makedir(cdir):
    try:
        os.makedirs(cdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def nll(params, chi2, l2, w):
    e_s = np.dot(l2, params)
    snp_ll = chi2/e_s + np.log(chi2) + np.log(2.*e_s) + np.log(np.pi)
    return np.dot(1./w, 0.5*snp_ll)


def compute_annot_overlap_matrix(annot_file_struct, frq_file_struct=None):

    annot_dfs = []

    for chr_num in range(22, 0, -1):
        adf = pd.read_csv(annot_file_struct % chr_num, sep="\s+")
        if frq_file_struct is not None and count_file == '.M_5_50':
            frq_df = pd.read_csv(frq_file_struct % chr_num, sep="\s+")
            adf = adf.loc[adf['SNP'].isin(frq_df.loc[frq_df['MAF'] > 0.05, 'SNP']), ]

        adf = adf.iloc[:, 4:]
        if keep_annotations is not None:
            adf = adf.loc[:, keep_annotations]

        annot_dfs.append(adf)

    annots = pd.concat(annot_dfs)

    annots_std = annots.std().values
    binary_annots = [c for c in annots.columns
                     if len(np.unique(annots[c])) == 2]

    annots = annots.values

    return np.dot(annots.T, annots), len(annots), binary_annots, annots_std


def read_merge_ldscore_files(ldf, wldf, ld_col):

    ref_df = pd.read_csv(ldf, sep="\t")
    ref_df = ref_df.sort_values(by='BP')
    M = pd.read_csv(ldf.replace(".ldscore.gz", count_file), header=None, sep="\t").values

    ld_dfs = {}

    for ldc in ld_col:

        m_ref_df = ref_df[['SNP'] + [c for c in ref_df.columns if c[-len(ldc):] == ldc]]

        if keep_annotations is not None:
            keep_idx = np.array([i for i, c in enumerate(m_ref_df.columns)
                                 if c == 'SNP' or c[:-len(ldc)] in keep_annotations])
            m_ref_df = m_ref_df.iloc[:, keep_idx]

        # The LD Score Weights file:
        w_df = pd.read_csv(wldf, sep="\t")

        try:
            w_df = w_df[['SNP', 'base' + ldc]]
            w_df = w_df.rename(columns={'base' + ldc: 'w_base' + ldc})
        except Exception as e:
            w_df = w_df[['SNP', 'L2']]
            w_df = w_df.rename(columns={'L2': 'w_base' + ldc})

        ld_dfs[ldc] = pd.merge(m_ref_df, w_df, on='SNP')

    if keep_annotations is not None:
        M = M[:, keep_idx[1:] - 1]

    return ld_dfs, M


def read_ld_scores(ref_ld_file_struct, w_ld_file_struct, ld_col):

    args = []

    if isinstance(ld_col, list):
        ld_col = ld_col
    else:
        ld_col = [ld_col]

    for chr_num in range(1, 23):
        args.append((ref_ld_file_struct % chr_num,
                     w_ld_file_struct % chr_num,
                     ld_col))

    pool = Pool(num_procs)
    res = pool.starmap(read_merge_ldscore_files, args)
    pool.close()
    pool.join()

    M = sum([r[1] for r in res])

    final_df = {}

    for ldc in ld_col:
        final_df[ldc] = pd.concat([r[0][ldc] for r in res])

    return final_df, M


def plot_regression_result(ldsc_colname, ldsc_w_colname, reg_res, M, data_df):

    fig = plt.figure(figsize=(10, 8))

    w_ld = np.maximum(data_df[ldsc_w_colname], 1.)
    data_df['wLd'] = 1. / w_ld

    data_df['binLd'] = pd.qcut(data_df[ldsc_colname], 100,
                               labels=np.quantile(data_df[ldsc_colname], np.linspace(0.0, 1., 100)))

    gdf = data_df.groupby('binLd', as_index=False).mean()

    sns.scatterplot(x='binLd', y='chi2', data=gdf, hue='wLd',
                    palette='ch:.25', zorder=10)

    x = np.linspace(0, max(gdf['binLd']), 1000)
    for reg_name, reg_val in reg_res.items():
        label = reg_name + ": " + ("$h_g^2=$%.3f, $b_0=$%.3f" % tuple(reg_val))
        plt.plot(x, N*reg_val[0]*x/M + reg_val[1], label=label)

    plt.xlabel("LD Score bin")
    plt.ylabel("Mean $\chi^2$")
    plt.title(ldsc_colname)

    plt.xlim(0., np.percentile(gdf['binLd'], 95) + 0.1 * np.percentile(gdf['binLd'], 95))
    plt.ylim(min(gdf['chi2']) - .1 * min(gdf['chi2']), max(gdf['chi2']) + .1 * max(gdf['chi2']))

    leg = plt.legend()

    for i, tex in enumerate(leg.get_texts()):
        if tex.get_text() == "wLd":
            tex.set_text("Regression weight")
        try:
            tex.set_text(str(round(float(tex.get_text()), 2)))
        except Exception as e:
            continue

    plt.savefig("./plots/regression/scz/" + ldsc_colname + ".png")


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


def plot_overall_metric_estimates(regression_results, metric='hg2'):

    methods = sorted(list(set([k.replace("-annot", "") for k in regression_results.keys()])))
    annotated_mean, annotated_yerr, baseline_mean, baseline_yerr = [], [], [], []

    for m in methods:
        annotated_mean.append(regression_results[m + '-annot'][metric])
        annotated_yerr.append(regression_results[m + '-annot'][metric + '_se'])

        baseline_mean.append(regression_results[m][metric])
        baseline_yerr.append(regression_results[m][metric + '_se'])

    x = np.arange(len(methods))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - width / 2, baseline_mean, width, yerr=baseline_yerr, label='w/o Annotations')
    rects2 = ax.bar(x + width / 2, annotated_mean, width, yerr=annotated_yerr, label='w/ Annotations')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Estimated ' + metric)
    ax.set_title('Estimated ' + metric + ' (' + trait_name + ')')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "total_" + metric + ".pdf"))
    plt.close()


def plot_annot_specific_results(regression_results, metric, annot_key='Annotations'):

    methods = sorted(list(set([k.replace("-annot", "") for k in regression_results.keys()])))

    annotations = np.array(regression_results[methods[0] + '-annot'][annot_key]['Names'])

    if 'tau' not in metric:
        keep_index = np.argwhere(np.isin(annotations, binary_annots)).ravel()
    else:
        keep_index = None

    annotations = annotations[keep_index].ravel()

    sorted_idx = np.argsort(np.abs(regression_results['SLDSC-annot']
                                   [annot_key][metric].ravel()[keep_index].ravel()))[::-1]
    annotations = annotations[sorted_idx]

    bar_vals = {}

    for m in methods:
        try:
            bar_vals[m] = {
                'mean': regression_results[m + '-annot'][annot_key][metric][keep_index].ravel()[sorted_idx],
                'se': regression_results[m + '-annot'][annot_key][metric + "_se"][keep_index].ravel()[sorted_idx]
            }
        except Exception as e:
            bar_vals[m] = {
                'mean': regression_results[m + '-annot'][annot_key][metric].ravel()[keep_index].ravel()[sorted_idx]
            }

    width = 0.4  # the width of the bars

    fig, axes = plt.subplots(figsize=(12, 6*int(np.ceil(len(annotations) / 10))),
                             nrows=int(np.ceil(len(annotations) / 10)), sharey=True)

    for xidx, ax in enumerate(axes):

        x = np.arange(len(annotations[10*xidx:10*(xidx + 1)]))*2
        rects = []

        for idx, m in enumerate(methods):
            if 'se' in bar_vals[m].keys():
                rects.append(ax.bar(x - (2 - idx)*width,
                                    bar_vals[m]['mean'][10*xidx:10*(xidx + 1)],
                                    width, yerr=bar_vals[m]['se'][10*xidx:10*(xidx + 1)], label=m))
            else:
                rects.append(ax.bar(x - (2 - idx)*width,
                                    bar_vals[m]['mean'][10*xidx:10*(xidx + 1)], width, label=m))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(x)
        ax.set_xticklabels(annotations[10*xidx:10*(xidx + 1)])
        ax.set_ylabel('Estimated ' + metric + ["", " (x$10^-5$)"][metric == 'tau'])

        if xidx == 0:
            ax.set_title('Estimated ' + metric + ' for standard annotations ' + trait_name)

        [autolabel(rect, ax) for rect in rects]

        for label in ax.get_xmajorticklabels():
            label.set_rotation(90)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, metric + "_" + annot_key + ".pdf"))
    plt.close()


# Read the LDSC files:
def get_baseline_ldscores():

    ld_ref_file = "./reference/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.%d.l2.ldscore.gz" #"./reference/baselineLD_v1.1/baselineLD.%d.l2.ldscore.gz" #
    ld_w_file = "./reference/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.%d.l2.ldscore.gz"

    ldsc, ldsc_M = read_ld_scores(ld_ref_file, ld_w_file, "L2")
    ldsc = ldsc['L2']

    ld_scores = []

    for w_annot in (True, False):
        if w_annot:
            c_ldsc = ldsc
            c_ldsc_M = ldsc_M
        else:
            c_ldsc = ldsc[['SNP', 'baseL2', 'w_baseL2']]
            c_ldsc_M= ldsc_M[:1, :1]

        ld_scores.append({
            'Name': 'SLDSC' + ['', '-annot'][w_annot],
            'Annotation': w_annot,
            'LDScores': c_ldsc,
            'Counts': c_ldsc_M,
            'WeightCol': 'w_baseL2',
            'Symbol': 'L2'
        })

    return ld_scores

# -----------------------------------------
# Read our LD score files:


def get_our_ldscores():

    ld_ref_file = "./output/Phase3_mldscores_MAF/EUR/D2_%d.l2.ldscore.gz"
    ld_w_file = "./output/Phase3_mldscores_MAF_weights/EUR/w_D2_%d.l2.ldscore.gz"

    our_scores, our_M = read_ld_scores(ld_ref_file, ld_w_file, ["LD2", "LD2MAF", "L2"])

    our_ld2 = our_scores["LD2"]
    our_ld2maf = our_scores["LD2MAF"]
    our_l2 = our_scores["L2"]

    ld_scores = []

    for w_annot in (True, False):

        if w_annot:
            c_our_ld2 = our_ld2
            c_our_ld2maf = our_ld2maf
            c_our_l2 = our_l2
            c_our_M = our_M
        else:
            c_our_ld2 = our_ld2[['SNP', 'baseLD2', 'w_baseLD2']]
            c_our_ld2maf = our_ld2maf[['SNP', 'baseLD2MAF', 'w_baseLD2MAF']]
            c_our_l2 = our_l2[['SNP', 'baseL2', 'w_baseL2']]
            c_our_M = our_M[:1, :1]

        ld_scores.append(
            {
                'Name': 'LD2' + ['', '-annot'][w_annot],
                'Annotation': w_annot,
                'LDScores': c_our_ld2,
                'Counts': c_our_M,
                'WeightCol': 'w_baseLD2',
                'Symbol': 'LD2'
            }
        )

        ld_scores.append(
            {
                'Name': 'LD2MAF' + ['', '-annot'][w_annot],
                'Annotation': w_annot,
                'LDScores': c_our_ld2maf,
                'Counts': c_our_M,
                'WeightCol': 'w_baseLD2MAF',
                'Symbol': 'LD2MAF'
            }
        )

        ld_scores.append(
            {
                'Name': 'Our L2' + ['', '-annot'][w_annot],
                'Annotation': w_annot,
                'LDScores': c_our_l2,
                'Counts': c_our_M,
                'WeightCol': 'w_baseL2',
                'Symbol': 'L2'
            }
        )

    return ld_scores


# -----------------------------------------
# Configurations:

num_procs = 7
count_file = '.M_5_50'
sumstats_dir = "/Users/szabad/Downloads/age_at_menarche.sumstats.gz" # "./data/independent_sumstats/UKB_460K.*.sumstats.gz" #
plot_dir = "./test_plots/regression/UKBB/%s/" % count_file.replace(".", "")
regression_dir = "./test_regression/UKBB/%s/" % count_file.replace(".", "")
keep_annotations = None

"""    
[
    'base',
    'MAFbin1',
    'MAFbin2',
    'MAFbin3',
    'MAFbin4',
    'MAFbin5',
    'MAFbin6',
    'MAFbin7',
    'MAFbin8',
    'MAFbin9',
    'MAFbin10'
]
"""

# -----------------------------------------

all_ld_scores = []
print("Reading Alkes LD Scores...")
all_ld_scores += get_alkes_ldscores()
print("Reading our LD Scores...")
#all_ld_scores += get_our_ldscores()

common_snps = list(set.intersection(*map(set, [ldc['LDScores']['SNP'] for ldc in all_ld_scores])))


print("Computing annotation covariance...")

annot_cov, annot_count, binary_annots, annots_std = compute_annot_overlap_matrix(
    #"./reference/baselineLD_v1.1/baselineLD.%d.annot.gz",
    "./reference/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.%d.annot.gz",
    "./data/1000G_Phase3_frq/1000G.EUR.QC.%d.frq")

# -----------------------------------------
# Perform regressions:

for ssf in glob.glob(sumstats_dir):

    trait_name = os.path.basename(ssf).replace(".sumstats.gz", "").replace("PASS_", "").replace("UKB_460K.", "")
    output_dir = os.path.join(plot_dir, trait_name)

    makedir(output_dir)
    makedir(regression_dir)

    ss_df = pd.read_csv(ssf, sep="\t")
    ss_df['CHISQ'] = ss_df['Z']**2

    #ss_df = ss_df.loc[ss_df['CHISQ'] <= max(80., 0.001*ss_df['N'][0]), ]

    print("Performing Regression on " + trait_name + "...")
    N = ss_df['N'][0]
    print("# Common SNPs:", len(common_snps))
    print("Sample size:", N)

    for ldc in all_ld_scores:

        print("---------")
        print(ldc['Name'])
        nss_df = pd.merge(ldc['LDScores'], ss_df)
        nss_df = nss_df.loc[nss_df['SNP'].isin(common_snps)]
        print("# of SNPs after merging:", len(nss_df))
        print("Mean Chi2:", np.mean(nss_df['CHISQ']))
        ld_score_names = [c for c in ldc['LDScores'].columns if ldc['Symbol'] in c and c != ldc['WeightCol']]
        x = nss_df[ld_score_names]

        reg = Hsq(nss_df[['CHISQ']].values,
                  x.values,
                  nss_df[[ldc['WeightCol']]].values,
                  nss_df[['N']].values,
                  ldc['Counts'],
                  old_weights=True)

        ldc['Regression'] = {
            'hg2': reg.tot,
            'hg2_se': reg.tot_se,
            'Intercept': reg.intercept,
            'Intercept_se': reg.intercept_se
        }

        print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
        print("total scores:", ldc['Regression'])
        print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")

        if ldc['Annotation']:
            """
            ldc['Regression']['Annotations'] = {
                'Names': [ln.replace(ldc['Symbol'], '') for ln in ld_score_names],
                'hg2': reg.cat.clip(min=0.),
                'hg2_se': reg.cat_se,
                'enrichment': reg.enrichment.clip(min=0.),
                'tau': reg.coef,
                'tau_se': reg.coef_se
            }
            """

            # TODO: Extend above code to include overlapping annotation computation.

            overlap_annot = reg._overlap_output(
                ld_score_names, annot_cov, ldc['Counts'], annot_count, True)

            coeff_factor = annots_std*annot_count/reg.tot
            
            tprob = 2.*stats.norm.sf(abs(overlap_annot['Coefficient_z-score'].values))
            tprob[tprob == 0.] = np.nan
            tprob = -np.log10(tprob)

            ldc['Regression']['OverlapAnnotations'] = {
                'Names': [ln.replace(ldc['Symbol'], '') for ln in ld_score_names],
                'hg2': overlap_annot['Prop._h2'].values.clip(min=0.),
                'hg2_se': overlap_annot['Prop._h2_std_error'].values,
                'enrichment': overlap_annot['Enrichment'].values.clip(min=0.),
                'enrichment_se': overlap_annot['Enrichment_std_error'].values,
                'enrichment_pvalue': -np.log10(pd.to_numeric(overlap_annot['Enrichment_p'].values,
                                                             errors='coerce')),
                'tau': overlap_annot['Coefficient'].values/1e-5,
                'tau_se': overlap_annot['Coefficient_std_error'].values/1e-5,
                'tau_pvalue': tprob,
                'tau_zscore': overlap_annot['Coefficient_z-score'].values,
                'tau_star': overlap_annot['Coefficient'].values*coeff_factor
            }

    final_reg_results = dict([(ldc['Name'], ldc['Regression']) for ldc in all_ld_scores])

    with open(os.path.join(regression_dir, trait_name + ".pickle"), 'wb') as handle:
        pickle.dump(final_reg_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    with open(os.path.join(regression_dir, trait_name + ".pickle"), 'rb') as handle:
        final_reg_results = pickle.load(handle)
    """

    plot_overall_metric_estimates(final_reg_results)
    plot_overall_metric_estimates(final_reg_results, metric="Intercept")

    """
    plot_annot_specific_results(final_reg_results, "hg2")
    plot_annot_specific_results(final_reg_results, "tau")
    plot_annot_specific_results(final_reg_results, "enrichment")
    """

    plot_annot_specific_results(final_reg_results, "hg2", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau_zscore", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau_pvalue", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau_star", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "enrichment", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "enrichment_pvalue", annot_key='OverlapAnnotations')
