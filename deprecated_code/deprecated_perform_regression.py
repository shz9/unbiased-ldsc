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
from collections.abc import Iterable
from scipy.optimize import curve_fit
import torch
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from gamma_glm_model import fit_ldscore_data
from multiprocessing import Pool
from ldsc.ldscore.regressions import Hsq

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def makedir(cdir):
    try:
        os.makedirs(cdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def compute_annot_overlap_matrix(annot_file_struct, frq_file_struct=None):

    annot_dfs = []

    maf_normalized_annot = 0.
    snps_per_maf_bin = [[] for _ in range(10)]

    for chr_num in range(22, 0, -1):
        adf = pd.read_csv(annot_file_struct % chr_num, sep="\s+")

        if frq_file_struct is not None:
            frq_df = pd.read_csv(frq_file_struct % chr_num, sep="\s+")

            if count_file == '.M_5_50':
                frq_df = frq_df.loc[frq_df['MAF'] > 0.05, ]

            frq_df['MAFVAR'] = 2.*frq_df['MAF']*(1. - frq_df['MAF'])

            adf = adf.loc[adf['SNP'].isin(frq_df['SNP']), ]

            mna_df = pd.merge(frq_df[['SNP', 'MAFVAR']], adf.iloc[:, [2] + list(range(4, len(adf.columns)))])

            snps_per_maf_bin = [s + list(mna_df.loc[mna_df['MAFbin' + str(i)] == 1, 'SNP'].values)
                                for i, s in enumerate(snps_per_maf_bin, 1)]

            maf_normalized_annot += np.dot(mna_df.iloc[:, 2:].values.T, mna_df['MAFVAR'].values)

        annot_dfs.append(adf.iloc[:, 4:])

    annots = pd.concat(annot_dfs)

    if keep_annotations is not None:
        maf_normalized_annot = maf_normalized_annot[np.isin(annots.columns, keep_annotations)]
        annots = annots.loc[:, keep_annotations]
    elif exclude_annotations is not None:
        maf_normalized_annot = maf_normalized_annot[~np.isin(annots.columns, exclude_annotations)]
        annots = annots.drop(exclude_annotations, axis=1)

    annots_std = annots.std().values
    binary_annots = [c for c in annots.columns
                     if len(np.unique(annots[c])) == 2]

    print("MAF normalized annots:", list(zip(annots.columns, maf_normalized_annot)))

    annots = annots.values

    return np.dot(annots.T, annots), len(annots), binary_annots, annots_std, maf_normalized_annot, snps_per_maf_bin


def read_merge_ldscore_files(ldf, wldf, ld_col):

    ref_df = pd.read_csv(ldf, sep="\t")
    M = pd.read_csv(ldf.replace(".ldscore.gz", count_file), header=None, sep="\t").values

    ld_dfs = {}

    for ldc in ld_col:

        ldc_cols = [c for c in ref_df.columns if c[-len(ldc):] == ldc]
        m_ref_df = ref_df[['CHR', 'SNP'] + ldc_cols]

        if keep_annotations is not None:
            keep_idx = np.array([i for i, c in enumerate(m_ref_df.columns)
                                 if c in ['CHR', 'SNP'] or c[:-len(ldc)] in keep_annotations])
            m_ref_df = m_ref_df.iloc[:, keep_idx]
        elif exclude_annotations is not None:
            keep_idx = np.array([i for i, c in enumerate(m_ref_df.columns)
                                 if c[:-len(ldc)] not in exclude_annotations])
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

    if keep_annotations is not None or exclude_annotations is not None:
        M = M[:, keep_idx[1:] - 2]

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

        try:
            annotated_yerr.append(regression_results[m + '-annot'][metric + '_se'])
        except KeyError:
            annotated_yerr.append(None)

        baseline_mean.append(regression_results[m][metric])

        try:
            baseline_yerr.append(regression_results[m][metric + '_se'])
        except KeyError:
            baseline_yerr.append(None)

    x = np.arange(len(methods))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 7))

    if any([y is None for y in baseline_yerr]):
        rects1 = ax.bar(x - width / 2, baseline_mean, width, label='w/o Annotations')
    else:
        rects1 = ax.bar(x - width / 2, baseline_mean, width, yerr=baseline_yerr, label='w/o Annotations')

    if any([y is None for y in annotated_yerr]):
        rects2 = ax.bar(x + width / 2, annotated_mean, width, label='w/ Annotations')
    else:
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

    if not isinstance(axes, Iterable):
        axes = [axes]

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


def plot_observed_vs_predicted_chi2(pred_df):

    for m in pred_df.columns:
        if m not in ['SNP', 'CHISQ']:
            plt.scatter(pred_df['CHISQ'], pred_df[m], label=m)

    x = np.linspace(0.0, max(pred_df['CHISQ']) + .1*max(pred_df['CHISQ']), 1000)
    plt.plot(x, x, linestyle='--', c='grey')
    plt.xlim([0., 100.])
    plt.ylim([0., 100.])
    plt.xlabel("Observed $\chi^2$")
    plt.ylabel("Predicted $\chi^2$")

    plt.legend()
    plt.savefig(os.path.join(output_dir, "observed_vs_predicted_chisq.png"))
    plt.close()


# Read the LDSC files:
def get_baseline_ldscores():

    ld_ref_file = "./reference/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.%d.l2.ldscore.gz"
    ld_w_file = "./reference/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.%d.l2.ldscore.gz"

    ldsc, ldsc_M = read_ld_scores(ld_ref_file, ld_w_file, "L2")
    ldsc = ldsc['L2']

    ld_scores = []

    for w_annot in (True, False):
        if w_annot:
            c_ldsc = ldsc
            c_ldsc_M = ldsc_M
        else:
            c_ldsc = ldsc[['CHR', 'SNP', 'baseL2', 'w_baseL2']]
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

    ld_ref_file = "./output/Phase3_mldscores_MAF_updated/EUR/D2_%d.l2.ldscore.gz"
    ld_w_file = "./output/Phase3_mldscores_MAF_weights/EUR/w_D2_%d.l2.ldscore.gz"

    # "LD2MAF", "L2"
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
            c_our_ld2 = our_ld2[['CHR', 'SNP', 'baseLD2', 'w_baseLD2']]
            c_our_ld2maf = our_ld2maf[['CHR', 'SNP', 'baseLD2MAF', 'w_baseLD2MAF']]
            c_our_l2 = our_l2[['CHR', 'SNP', 'baseL2', 'w_baseL2']]
            c_our_M = our_M[:1, :1]

        ld_scores.append(
            {
                'Name': 'LD2' + ['', '-annot'][w_annot],
                'Annotation': w_annot,
                'LDScores': c_our_ld2,
                'Counts': np.copy(c_our_M),
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


class LDScoreData(Dataset):
    def __init__(self, X, Y, N, w=None):

        self.x = torch.ones((X.shape[0], X.shape[1] + 1)) / N
        self.x[:, :-1] = torch.from_numpy(X)
        self.y = torch.from_numpy(Y)
        if w is None:
            self.w = torch.ones(len(self.x))
        else:
            self.w = torch.from_numpy(w)

        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.w[index]

    def __len__(self):
        return self.len


def loglikelihood(chi2, pred_chi2, w, null_pred_chi2=None):
    """
    * chi2: marginal chi2 from GWAS
    * pred_chi2: predicted chi2 from fitted model
    * w: weights
    * null_pred_chi2: predicted chi2 under the null model
    """

    print("1/w stats:", stats.describe(1. / w))
    print("num zeros:", (pred_chi2 <= 0.).sum())
    print("total number:", len(pred_chi2))
    print("contribution to log-likelihood:", (pred_chi2 <= 0.).sum() * np.log(2. * 1e-6))

    # Set non-positive chi2/pred chi2 to 1e-6
    chi2[chi2 <= 0.] = 1e-6
    pred_chi2[pred_chi2 <= 0.] = 1e-6

    # Compute the loglikelihood for each SNP:
    snp_ll = chi2/pred_chi2 + np.log(chi2) + np.log(2.*pred_chi2) + np.log(np.pi)
    print("func ll stats:", stats.describe(-.5*snp_ll))
    # Weighted sum of SNP log-likelihoods:
    func_ll = np.dot(1./w, -0.5*snp_ll)

    if null_pred_chi2 is None:
        return func_ll

    else:
        null_pred_chi2[null_pred_chi2 <= 0.] = 1e-6

        null_snp_ll = chi2 / null_pred_chi2 + np.log(chi2) + np.log(2. * null_pred_chi2) + np.log(np.pi)
        print("nill ll stats:", stats.describe(-.5*null_snp_ll))
        null_ll = np.dot(1./w, -0.5*null_snp_ll)

        return -2.*(null_ll - func_ll)


def predict_chi2(params, inputs, N):
    coef, intercept = params
    return N*np.dot(inputs, coef) + intercept


def negative_loglikelihood(params, inputs, targets, N, w=None):

    pred = predict_chi2(params, inputs, N)
    snp_ll = targets / pred + np.log(targets) + np.log(2. * pred) + np.log(np.pi)

    if w is None:
        w = np.ones(len(snp_ll))

    return np.dot(1./w, snp_ll)


def torch_negative_loglikelihood(params, inputs, targets, N, w):

    pred = torch.clamp(N*torch.matmul(inputs, params), min=1e-6)
    targets = torch.clamp(targets, min=1e-6)
    snp_ll = targets / pred + torch.log(targets) + torch.log(2. * pred) + np.log(np.pi)
    return torch.matmul((1./w).T, snp_ll)


def weighted_cov(chi2, pred_chi2, w):
    return np.dot(1./w, chi2*pred_chi2)*np.sum(1./w) - np.dot(1./w, chi2)*np.dot(1./w, pred_chi2)

# -----------------------------------------
# Configurations:

batch_size = 512
num_procs = 6
torch_optimize = False
count_file = '.M_5_50'
sumstats_dir = "./data/independent_sumstats/UKB_460K.body_H*.sumstats.gz" #
plot_dir = "./plots/regression/UKBB_MAFbins/%s/" % count_file.replace(".", "")
regression_dir = "./regression/UKBB_MAFbins/%s/" % count_file.replace(".", "")
exclude_annotations = None #['MAF_Adj_ASMC']
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
    'MAFbin10',
    'MAF_Adj_ASMC'
]
"""
# -----------------------------------------

print("Computing annotation covariance...")

"""
annot_cov, annot_count, binary_annots, annots_std, maf_normalized_annot, snps_per_bin = compute_annot_overlap_matrix(
    #"./reference/baselineLD_v1.1/baselineLD.%d.annot.gz",
    "./reference/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.%d.annot.gz",
    "./data/1000G_Phase3_frq/1000G.EUR.QC.%d.frq")

"""

with open("./annot_cov " + count_file + ".pickle", 'rb') as handle:
    annot_cov, annot_count, binary_annots, annots_std, maf_normalized_annot, snps_per_bin = pickle.load(handle)


"""
with open("./annot_cov " + count_file + ".pickle", 'wb') as handle:
    pickle.dump((annot_cov, annot_count, binary_annots, annots_std, maf_normalized_annot, snps_per_bin),
                handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

all_ld_scores = []
print("Reading Alkes LD Scores...")
all_ld_scores += get_baseline_ldscores()
print("Reading our LD Scores...")
all_ld_scores += get_our_ldscores()

common_snps = list(set.intersection(*map(set, [ldc['LDScores']['SNP'] for ldc in all_ld_scores])))

# -----------------------------------------
# Perform regressions:

for ssf in glob.glob(sumstats_dir):

    predicted_chi2 = []

    trait_name = os.path.basename(ssf).replace(".sumstats.gz", "").replace("PASS_", "").replace("UKB_460K.", "")
    output_dir = os.path.join(plot_dir, trait_name)

    makedir(output_dir)
    makedir(regression_dir)

    ss_df = pd.read_csv(ssf, sep="\t")
    ss_df['CHISQ'] = ss_df['Z']**2

    ss_df = ss_df.loc[ss_df['CHISQ'] <= max(80., 0.001*ss_df['N'][0]), ]

    print("Performing Regression on " + trait_name + "...")
    N = ss_df['N'][0]
    print("# Common SNPs:", len(common_snps))
    print("Sample size:", N)

    for ldc in all_ld_scores:

        print("---------")
        print(ldc['Name'])
        nss_df = pd.merge(ldc['LDScores'], ss_df)
        nss_df = nss_df.loc[nss_df['SNP'].isin(common_snps)]
        #print("Counts:", ldc['Counts'])
        print("# of SNPs after merging:", len(nss_df))
        print("Mean Chi2:", np.mean(nss_df['CHISQ']))
        ld_score_names = [c for c in ldc['LDScores'].columns if ldc['Symbol'] in c and c != ldc['WeightCol']]
        #if 'LD2-annot' in ldc['Name']:
            #ldc['Counts'][:, [c[:-len('LD2')] not in binary_annots + ['base'] for c in ld_score_names]] = 1.
        #    print("Updated counts:", list(zip(ld_score_names, ldc['Counts'][0])))
        if ldc['Symbol'][-len(ldc['Name']):] == 'LD2':
            ldc['Counts'] = np.array([list(maf_normalized_annot[:len(ld_score_names)])])
            #x = (nss_df[ld_score_names]/maf_normalized_annot[:len(ld_score_names)])*ldc['Counts']

        #if 'annot' in ldc['Name']:
        #    print("ASMC:", x[['MAFbin5' + ldc['Symbol'], 'MAF_Adj_ASMC' + ldc['Symbol']]])

        nss_df.to_csv(ldc['Name'] + ".csv")

        """
        if torch_optimize:
            ldc_data = LDScoreData(nss_df[ld_score_names].values,
                                   nss_df[['CHISQ']].values,
                                   np.mean(nss_df[['N']].values),
                                   nss_df[[ldc['WeightCol']]].values)
            data_loader = DataLoader(dataset=ldc_data, batch_size=batch_size)

            params = torch.zeros((len(ld_score_names) + 1, 1))
            params[-1, 0] = 1.

            params.requires_grad_()
            optimizer = torch.optim.Adam([params], 1e-8)

            for ii in range(20):
                for X, Y, W in data_loader:
                    loss = torch_negative_loglikelihood(params, X, Y, N, W)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if ii % 10 == 0:
                    print('Step # {}, loss: {}'.format(ii, loss))
        """

        reg = Hsq(nss_df[['CHISQ']].values,
                  nss_df[ld_score_names].values,
                  nss_df[[ldc['WeightCol']]].values,
                  nss_df[['N']].values,
                  ldc['Counts'],
                  old_weights=True)

        ldc['Regression'] = {
            'hg2': reg.tot,
            'hg2_se': reg.tot_se,
            'Intercept': reg.intercept,
            'Intercept_se': reg.intercept_se,
            'Coefficients': list(zip(ld_score_names, reg.coef))
        }

        pred_chi2 = nss_df['N'][0]*np.dot(nss_df[ld_score_names].values, reg.coef) + reg.intercept
        pred_df = nss_df[['SNP', 'CHISQ']]
        pred_df[ldc['Name']] = pred_chi2
        predicted_chi2.append(pred_df)

        #ldc['Regression']['LL'] = loglikelihood(nss_df['CHISQ'].values, pred_chi2, nss_df[ldc['WeightCol']].values,
        #                                        null_pred_chi2=np.repeat(reg.intercept, len(nss_df)))

        ldc['Regression'].update(fit_ldscore_data(nss_df, ld_score_names, ldc['WeightCol'], link='log'))
        print("finished gamma regression...")

        """
        rho = []
        w_rho = []
        mse = []

        maf_rho = [[] for _ in range(10)]
        maf_mse = [[] for _ in range(10)]

        for chr_num in range(1, 23):

            train_nss_df = nss_df.loc[nss_df['CHR'] != chr_num, ]
            test_nss_df = nss_df.loc[nss_df['CHR'] == chr_num, ]

            train_reg = Hsq(train_nss_df[['CHISQ']].values,
                            train_nss_df[ld_score_names].values,
                            train_nss_df[[ldc['WeightCol']]].values,
                            train_nss_df[['N']].values,
                            ldc['Counts'],
                            old_weights=True)

            test_pred_chi2 = nss_df['N'][0] * np.dot(test_nss_df[ld_score_names].values,
                                                     train_reg.coef) + train_reg.intercept

            rho.append(np.corrcoef(test_pred_chi2, test_nss_df['CHISQ'].values)[0, 1])
            mse.append(np.mean((test_pred_chi2 - test_nss_df['CHISQ'].values)**2))

            for i in range(10):
                maf_test_nss_df = test_nss_df.loc[test_nss_df['SNP'].isin(snps_per_bin[i]), ]
                if len(maf_test_nss_df) > 0:
                    maf_test_pred_chi2 = nss_df['N'][0] * np.dot(maf_test_nss_df[ld_score_names].values,
                                                                 train_reg.coef) + train_reg.intercept

                    maf_rho[i].append(np.corrcoef(maf_test_pred_chi2, maf_test_nss_df['CHISQ'].values)[0, 1])
                    maf_mse[i].append(np.mean((maf_test_pred_chi2 - maf_test_nss_df['CHISQ'].values)**2))
                else:
                    maf_rho[i].append(np.nan)
                    maf_mse[i].append(np.nan)

            try:
                w_corr = weighted_cov(test_pred_chi2,
                                      test_nss_df['CHISQ'].values,
                                      test_nss_df[ldc['WeightCol']].values) / \
                         np.sqrt(weighted_cov(test_nss_df['CHISQ'].values,
                                              test_nss_df['CHISQ'].values,
                                              test_nss_df[ldc['WeightCol']].values) *
                                 weighted_cov(test_pred_chi2,
                                              test_pred_chi2,
                                              test_nss_df[ldc['WeightCol']].values))
            except Exception:
                w_corr = np.nan

            w_rho.append(w_corr)

        ldc['Regression']['rho'] = np.mean(rho)
        ldc['Regression']['rho_se'] = np.std(rho) / np.sqrt(22)
        ldc['Regression']['MSE'] = np.mean(mse)
        ldc['Regression']['MSE_se'] = np.std(mse) / np.sqrt(22)
        ldc['Regression']['w_rho'] = np.nanmean(w_rho)
        ldc['Regression']['w_rho_se'] = np.nanstd(w_rho) / np.sqrt(22)

        for i in range(10):
            ldc['Regression']['MSE_MAFbin' + str(i + 1)] = np.nanmean(maf_mse[i])
            ldc['Regression']['MSE_MAFbin' + str(i + 1) + '_se'] = np.nanstd(maf_mse[i])

            ldc['Regression']['rho_MAFbin' + str(i + 1)] = np.nanmean(maf_rho[i])
            ldc['Regression']['rho_MAFbin' + str(i + 1) + '_se'] = np.nanstd(maf_rho[i])
        """

        print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")
        print("Counts:", list(zip(ld_score_names, ldc['Counts'][0])))
        print("\/\/\/\/\/\/\/\/\/----------------\/\/\/\/\/\/\/\/\/\/")
        print("total scores:", ldc['Regression'])
        print("\/\/\/\/\/\/\/\/\/----------------\/\/\/\/\/\/\/\/\/\/")
        print("unnormalized prop", list(zip(ld_score_names, reg.prop[0])))
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
            print("annot_count", annot_count)
            overlap_annot = reg._overlap_output(
                ld_score_names, annot_cov, ldc['Counts'], annot_count, True)

            coeff_factor = annots_std*annot_count/reg.tot
            
            tprob = 2.*stats.norm.sf(abs(overlap_annot['Coefficient_z-score'].values))
            tprob[tprob == 0.] = np.nan
            tprob = -np.log10(tprob)

            print("normalized prop", list(zip(ld_score_names, overlap_annot['Prop._h2'].values)))
            print("\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/")

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

    #final_pred_results = reduce(lambda x, y: pd.merge(x, y, on=['SNP', 'CHISQ']), predicted_chi2)
    #final_pred_results.to_csv("./pred_results.csv")
    #print("final_pred_results:", final_pred_results.head())
    #plot_observed_vs_predicted_chi2(final_pred_results)

    with open(os.path.join(regression_dir, trait_name + ".pickle"), 'wb') as handle:
        pickle.dump(final_reg_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """
    with open(os.path.join(regression_dir, trait_name + ".pickle"), 'rb') as handle:
        final_reg_results = pickle.load(handle)
    """
    print("Plotting...")
    plot_overall_metric_estimates(final_reg_results)
    plot_overall_metric_estimates(final_reg_results, metric="Intercept")
    plot_overall_metric_estimates(final_reg_results, metric="LL")
    plot_overall_metric_estimates(final_reg_results, metric="AIC")
    plot_overall_metric_estimates(final_reg_results, metric="BIC")
    plot_overall_metric_estimates(final_reg_results, metric="LRT")
    plot_overall_metric_estimates(final_reg_results, metric="adjR2")

    """
    plot_overall_metric_estimates(final_reg_results, metric="rho")
    plot_overall_metric_estimates(final_reg_results, metric="w_rho")
    plot_overall_metric_estimates(final_reg_results, metric="MSE")

    for i in range(10):
        plot_overall_metric_estimates(final_reg_results, metric='MSE_MAFbin' + str(i + 1))
        plot_overall_metric_estimates(final_reg_results, metric='rho_MAFbin' + str(i + 1))
    """

    plot_annot_specific_results(final_reg_results, "hg2", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau_zscore", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau_pvalue", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "tau_star", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "enrichment", annot_key='OverlapAnnotations')
    plot_annot_specific_results(final_reg_results, "enrichment_pvalue", annot_key='OverlapAnnotations')
