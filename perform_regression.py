import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
import copy
import argparse
from itertools import product
from gamma_glm_model import get_model_lrt
from multiprocessing import Pool
from ldsc.ldscore.regressions import Hsq
from utils import write_pbz2, read_pbz2


def bin_regression_df(reg_df, ld_colname, w_ld_colname):
    """
    This function bins the regression dataframe to 100 quantiles.
    This facilitates producing plots similar to original figures
    in Bulik-Sullivan et al. 2015.
    """

    w_ld = np.maximum(reg_df[w_ld_colname], 1.)
    reg_df['wLD'] = 1. / w_ld

    reg_df['binLD'] = pd.qcut(reg_df[ld_colname], 100,
                               labels=np.quantile(reg_df[ld_colname],
                                                  np.linspace(0.0, 1., 100)))

    gdf = reg_df.groupby('binLD', as_index=False).mean()

    return gdf


def compute_annot_stats(annot_file_struct, frq_file_struct,
                        maf_5_50=True, alpha=(0., .25, .5, .75, 1.)):

    """
    This function compute statistics for the model annotations,
    including covariance between the annotations, snps per maf bin,
    and sums and MAF-weighted sums of each annotation.
    """

    w_annot_dfs = []
    mafvar_dfs = []
    snps_per_maf_bin = [[] for _ in range(10)]

    for chr_num in range(22, 0, -1):

        # Read the annotation and frequency dataframes:
        adf = pd.read_csv(annot_file_struct % chr_num, sep="\s+")
        frq_df = pd.read_csv(frq_file_struct % chr_num, sep="\s+")

        if maf_5_50:
            frq_df = frq_df.loc[frq_df['MAF'] > 0.05, ]

        frq_df['MAFVAR'] = 2.*frq_df['MAF']*(1. - frq_df['MAF'])
        adf = adf.loc[adf['SNP'].isin(frq_df['SNP']), ]

        mna_df = pd.merge(frq_df[['SNP', 'MAFVAR']],
                          adf.iloc[:, [2] + list(range(4, len(adf.columns)))])

        snps_per_maf_bin = [s + list(adf.loc[adf['MAFbin' + str(i)] == 1, 'SNP'].values)
                            for i, s in enumerate(snps_per_maf_bin, 1)]

        w_annot_dfs.append(mna_df.iloc[:, 2:])
        mafvar_dfs.append(mna_df[['MAFVAR']])

    w_annots = pd.concat(w_annot_dfs)
    mafvars = pd.concat(mafvar_dfs)

    w_annots_std = {
        a: ((mafvars.values**(1. - a))*w_annots).std().values
        for a in alpha
    }

    w_annot_sum = {
        a: np.dot(w_annots.values.T, mafvars.values**(1. - a)).reshape(1, -1)
        for a in alpha
    }

    w_annot_cov = {
        a: np.dot((np.sqrt(mafvars.values ** (1. - a)) * w_annots).values.T,
                  (np.sqrt(mafvars.values ** (1. - a)) * w_annots).values)
        for a in alpha
    }

    binary_annots = [c for c in w_annots.columns
                     if len(np.unique(w_annots[c])) == 2]

    return {
        'Covariance': w_annot_cov,
        'M': len(w_annots),  # number of snps
        'Names': np.array(list(w_annots.columns)),
        'Binary annotations': binary_annots,
        'Annotation std': w_annots_std,
        'Annotation sum': w_annot_sum,
        'SNPs per MAF bin': snps_per_maf_bin
    }


def read_merge_ldscore_files(ldf,
                             wldf,
                             ld_col):

    ref_df = pd.read_csv(ldf, sep="\t")
    M = pd.read_csv(ldf.replace(".ldscore.gz", count_file),
                    header=None, sep="\t").values

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


def read_ld_scores_parallel(ref_ld_file_struct, w_ld_file_struct, ld_col):

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

# -----------------------------------------
# Read baseline LD score files:


def read_baseline_ldscores(
        ld_ref_file="data/ld_scores/1000G_Phase3_EUR_baselineLD_v2.2_ldscores/baselineLD.%d.l2.ldscore.gz",
        ld_w_file="data/ld_scores/1000G_Phase3_EUR_weights_hm3_no_MHC/weights.hm3_noMHC.%d.l2.ldscore.gz"):

    print("> Reading baseline LD Scores...")

    ldsc, ldsc_M = read_ld_scores_parallel(ld_ref_file, ld_w_file, "L2")
    ldsc = ldsc['L2']

    data = []

    for w_annot in (True, False):
        if w_annot:
            c_ldsc = ldsc
            c_ldsc_M = ldsc_M
        else:
            c_ldsc = ldsc[['CHR', 'SNP', 'baseL2', 'w_baseL2']]
            c_ldsc_M= ldsc_M[:1, :1]

        data.append({
            'Name': ['LDSC', 'S-LDSC'][w_annot],
            'Annotation': w_annot,
            'LDScores': c_ldsc,
            'Counts': c_ldsc_M,
            'WeightCol': 'w_baseL2',
            'Symbol': 'L2'
        })

    return data

# -----------------------------------------
# Read modified LD score files:


def read_modified_ldscores(
        ld_ref_file="output/ld_scores/1000G_Phase3_EUR_mldscores/{ldc}/LD.%d.l2.ldscore.gz",
        ld_w_file="output/ld_scores_weights/1000G_Phase3_EUR_mldscores/{ldc}/LD.%d.l2.ldscore.gz",
        ld_estimator=("R2", "D2"),
        alpha=(0., .25, .5, .75, 1.)):

    print("> Reading modified LD Scores...")
    data = []

    for lde, a in product(ld_estimator, alpha):

        score_symbol = f"{lde}_{a}"

        if score_symbol not in reg_ld_scores:
            continue

        modified_scores, modified_M = read_ld_scores_parallel(
            ld_ref_file.format(ldc=score_symbol),
            ld_w_file.format(ldc=score_symbol),
            score_symbol)

        modified_scores = modified_scores[score_symbol]

        for w_annot in (True, False):
            if w_annot:
                c_ldsc = modified_scores
                c_ldsc_M = modified_M
            else:
                c_ldsc = modified_scores[['CHR', 'SNP', 'base' + score_symbol, 'w_base' + score_symbol]]
                c_ldsc_M = modified_M[:1, :1]

            data.append({
                'Name': [score_symbol, 'S-' + score_symbol][w_annot],
                'Annotation': w_annot,
                'LDScores': c_ldsc,
                'Counts': c_ldsc_M,
                'WeightCol': 'w_base' + score_symbol,
                'Symbol': score_symbol,
                'alpha': a
            })

    return data


def weighted_cov(chi2, pred_chi2, w):
    """
    Used to compute weighted correlation as defined by Speed, Holmes and Balding
    https://www.biorxiv.org/content/10.1101/736496v2.full
    """
    return np.dot(1./w, chi2*pred_chi2)*np.sum(1./w) - np.dot(1./w, chi2)*np.dot(1./w, pred_chi2)


def predict_chi2(tau, intercept, ld_scores, N):
    return N*np.dot(ld_scores, tau) + intercept


def compute_prediction_metrics(pred_chi2, true_chi2, w):

    w = np.maximum(w, 1.)

    return {
        'Mean Predicted Chisq': np.mean(pred_chi2),
        'Mean Difference': np.mean(pred_chi2 - true_chi2),
        'Weighted Mean Difference': np.mean((1./w)*(pred_chi2 - true_chi2)),
        'Mean Squared Difference': np.mean((pred_chi2 - true_chi2)**2),
        'Weighted Squared Mean Difference': np.mean((1. / w) * (pred_chi2 - true_chi2)**2),
        'Correlation': np.corrcoef(pred_chi2, true_chi2)[0, 1],
        'Weighted Correlation': weighted_cov(true_chi2, pred_chi2, w) / np.sqrt(
            weighted_cov(true_chi2, true_chi2, w)*weighted_cov(pred_chi2, pred_chi2, w)
        )
    }

# -----------------------------------------
# Perform regressions:


def perform_ldsc_regression(ld_scores,
                            trait_info,
                            annot_data,
                            chi2_filter=True,
                            lds_filter=True):

    trait_name = trait_info['Trait name']
    trait_subdir = os.path.basename(trait_info['File']).replace('.sumstats.gz', '')

    output_dir = os.path.join(regres_dir, trait_subdir)
    res_cache_dir = os.path.join(cache_dir, trait_subdir)

    ss_df = pd.read_csv(trait_info['File'], sep="\t")
    ss_df['CHISQ'] = ss_df['Z']**2

    if chi2_filter:
        ss_df = ss_df.loc[ss_df['CHISQ'] <= max(80., 0.001*ss_df['N'][0]), ]

    print(f">>> Processing GWAS summary statistics for {trait_name}...")

    for ldc in ld_scores:

        print(f"> Regressing with {ldc['Name']}...")

        if lds_filter:
            base_scores = ldc['LDScores']['base' + ldc['Symbol']]
            mean_sc, std_sc = base_scores.mean(), base_scores.std()

            ldc['LDScores'] = ldc['LDScores'].loc[(base_scores-mean_sc).abs() <= 1.96*std_sc]

        nss_df = pd.merge(ldc['LDScores'], ss_df)
        nss_df = nss_df.loc[nss_df['SNP'].isin(common_snps)]

        ld_score_names = [c for c in ldc['LDScores'].columns
                          if ldc['Symbol'] in c and c != ldc['WeightCol']]

        reg_counts = annot_data['Annotation sum'][ldc['alpha']][:1, :len(ld_score_names)]

        if cache_reg_df:
            write_pbz2(os.path.join(res_cache_dir, f"{ldc['Name']}.pbz2"), (
                nss_df, ld_score_names, ldc['WeightCol'], reg_counts, ldc['Counts']
            ))

        reg = Hsq(nss_df[['CHISQ']].values,
                  nss_df[ld_score_names].values,
                  nss_df[[ldc['WeightCol']]].values,
                  nss_df[['N']].values,
                  reg_counts,
                  old_weights=True)

        bin_df = bin_regression_df(nss_df, 'base' + ldc['Symbol'], ldc['WeightCol'])

        ldc['Regression'] = {
            'method': ldc['Name'],
            'binned_dataframe': bin_df,
            'N': nss_df[['N']].values[0],
            'M': annot_data['M'],
            'Annotation Sum': reg_counts,
            'Counts': ldc['Counts'],
            'hg2': reg.tot,
            'hg2_se': reg.tot_se,
            'Mean Chi2': np.mean(nss_df['CHISQ']),
            'Intercept': reg.intercept,
            'Intercept_se': reg.intercept_se,
            'Ratio': reg.ratio,
            'Ratio_se': reg.ratio_se,
            'Coefficients': list(zip(ld_score_names, reg.coef))
        }

        ldc['Regression']['LRT'] = get_model_lrt(reg.coef, reg.intercept,
                                                 nss_df, ld_score_names, ldc['WeightCol'])
        ldc['Regression']['LRT_se'] = 0.0

        pred_chi2 = predict_chi2(reg.coef, reg.intercept,
                                 nss_df[ld_score_names].values, np.mean(nss_df[['N']].values))

        ldc['Regression']['Predictive Performance'] = {
            'Overall': compute_prediction_metrics(pred_chi2,
                                                  nss_df['CHISQ'],
                                                  nss_df[ldc['WeightCol']]),
            'Per MAF bin': {}
        }

        for i, mafbin in enumerate(annot_data['SNPs per MAF bin']):
            subset = nss_df['SNP'].isin(mafbin)
            ldc['Regression']['Predictive Performance']['Per MAF bin'][i + 1] = compute_prediction_metrics(
                pred_chi2[subset],
                nss_df.loc[subset, 'CHISQ'],
                nss_df.loc[subset, ldc['WeightCol']]
            )

        if ldc['Annotation']:

            cov_mat = copy.deepcopy(annot_data['Covariance'][ldc['alpha']])

            overlap_annot2 = reg._overlap_output(
                ld_score_names, cov_mat,
                reg_counts, reg_counts[0][0], True
            )

            for i in range(cov_mat.shape[0]):
                cov_mat[i, :] = ldc['Counts']*cov_mat[i, :] / reg_counts

            overlap_annot = reg._overlap_output(
                ld_score_names, cov_mat,
                ldc['Counts'], annot_data['M'], True
            )

            coeff_factor_orig = annot_data['Annotation std'][1.]*annot_data['M']/reg.tot

            coeff_factor_mod1 = annot_data['Annotation std'][ldc['alpha']] * annot_data['M'] / reg.tot
            coeff_factor_mod2 = annot_data['Annotation std'][1.]*reg_counts[0][0]/reg.tot
            
            tau_pval = 2.*stats.norm.sf(abs(overlap_annot['Coefficient_z-score'].values))
            tau_pval[tau_pval == 0.] = np.nan
            tau_pval = -np.log10(tau_pval)

            diff_enrichment = (
                    reg.cat[0] / ldc['Counts'][0] -
                    (reg.tot - reg.cat[0]) / (ldc['Counts'][0][0] - ldc['Counts'][0])
            )
            diff_enrichment_se = (
                np.abs(reg.cat_se / ldc['Counts'][0] -
                       (reg.tot - reg.cat_se) / (ldc['Counts'][0][0] - ldc['Counts'][0]))
            )

            ldc['Regression']['Annotations'] = {
                'Names': [ln.replace(ldc['Symbol'], '') for ln in ld_score_names],
                'hg2': overlap_annot['Prop._h2'].values.clip(min=0.),
                'hg2_se': overlap_annot['Prop._h2_std_error'].values,
                'enrichment': overlap_annot['Enrichment'].values.clip(min=0.),
                'enrichment_se': overlap_annot['Enrichment_std_error'].values,
                'enrichment2': overlap_annot2['Enrichment'].values.clip(min=0.),
                'enrichment2_se': overlap_annot2['Enrichment_std_error'].values,
                'differential_enrichment': diff_enrichment,
                'differential_enrichment_se': diff_enrichment_se,
                'tau': overlap_annot['Coefficient'].values,
                'tau_se': overlap_annot['Coefficient_std_error'].values,
                'tau_pvalue': tau_pval,
                'tau_zscore': overlap_annot['Coefficient_z-score'].values,
                'tau_star': overlap_annot['Coefficient'].values*coeff_factor_orig,
                'tau_star_se': overlap_annot['Coefficient_std_error'].values*coeff_factor_orig,
                'tau_star_w': overlap_annot['Coefficient'].values*coeff_factor_mod1,
                'tau_star_w_se': overlap_annot['Coefficient_std_error'].values*coeff_factor_mod1,
                'tau_star_w2': overlap_annot['Coefficient'].values*coeff_factor_mod2,
                'tau_star_w2_se': overlap_annot['Coefficient_std_error'].values*coeff_factor_mod2,
            }

            ldc['Regression']['Annotations']['Predictive Performance'] = {}

            for an, spn in snps_per_annotation.items():

                ann_subset = nss_df['SNP'].isin(spn)

                ldc['Regression']['Annotations']['Predictive Performance'][an] = {
                    'Overall': compute_prediction_metrics(pred_chi2[ann_subset],
                                                          nss_df.loc[ann_subset, 'CHISQ'],
                                                          nss_df.loc[ann_subset, ldc['WeightCol']]),
                    'Per MAF bin': {}
                }

                for i, mafbin in enumerate(annot_data['SNPs per MAF bin']):
                    maf_subset = nss_df['SNP'].isin(mafbin)
                    ldc['Regression']['Annotations']['Predictive Performance'][an]['Per MAF bin'][i + 1] = compute_prediction_metrics(
                        pred_chi2[ann_subset & maf_subset],
                        nss_df.loc[ann_subset & maf_subset, 'CHISQ'],
                        nss_df.loc[ann_subset & maf_subset, ldc['WeightCol']]
                    )

        write_pbz2(os.path.join(output_dir, f"{ldc['Name']}.pbz2"),
                   ldc['Regression'])


if __name__ == '__main__':
    # -----------------------------------------
    # Configurations:

    np.seterr(divide='warn')
    parser = argparse.ArgumentParser(description='LD Score Regression Using 1000 Genomes Project Data')
    parser.add_argument('--pop', dest='pop', type=str, default='EUR',
                        help='The reference population name')
    parser.add_argument('--ldscores', dest='ld_scores', type=str,
                        default='R2_1.0',
                        help='The LD Score to use in the regression')

    args = parser.parse_args()

    ref_pop = args.pop  # Reference population
    reg_ld_scores = args.ld_scores.split(',')
    cache_reg_df = False
    num_procs = 6

    filter_configs = [(True, True, False)]
    #filter_configs = product(*[[True, False]] * 3)

    # -----------------------------------------

    for maf_5_50_filter, chi2_filter, lds_filter in filter_configs:

        count_file = '.M' + ['', '_5_50'][maf_5_50_filter]

        dir_name_struct = (count_file.replace('.', '') +
                           ['', '_chi2filt'][chi2_filter] +
                           ['', '_ldsfilt'][lds_filter])

        print(dir_name_struct)

        # Reference data
        sumstats_dir = "data/independent_sumstats/"
        reference_annot_file = "data/ld_scores/1000G_Phase3_EUR_baselineLD_v2.2_ldscores/baselineLD.%d.annot.gz"
        reference_freq_file = f"data/genotype_files/1000G_Phase3_{ref_pop}_plinkfiles/1000G.{ref_pop}.QC.%d.frq"

        # Inputs and outputs:
        snps_per_annotation_file = "data/annotations/snps_per_annotation.pbz2"
        cache_dir = f"cache/regression/{dir_name_struct}/"
        sumstats_file = "data/independent_sumstats/sumstats_table.csv"
        annot_stats_file = f"data/annotations/annotation_data/{ref_pop}/{count_file.replace('.', '')}.pbz2"
        regres_dir = f"results/regression/{ref_pop}/{dir_name_struct}/"
        exclude_annotations = None
        keep_annotations = None

        # -----------------------------------------
        # Reading annotation statistics:

        if os.path.isfile(annot_stats_file) and os.path.isfile(snps_per_annotation_file):
            print("> Loading annotation statistics...")
            annot_data = read_pbz2(annot_stats_file)
            snps_per_annotation = read_pbz2(snps_per_annotation_file)
        else:
            print("> Computing annotation statistics...")
            annot_data = compute_annot_stats(reference_annot_file,
                                             reference_freq_file,
                                             maf_5_50=count_file == '.M_5_50')
            write_pbz2(annot_stats_file, annot_data)

        if exclude_annotations is not None:

            exclude_idx = [c not in exclude_annotations
                           for c in annot_data['Names']]

            annot_data['Covariance'] = annot_data['Covariance'][exclude_idx, exclude_idx]
            annot_data['MAF Normalized sum'] = annot_data['MAF Normalized sum'][exclude_idx]
            annot_data['Annotation std'] = annot_data['Annotation std'][exclude_idx]
            annot_data['Weighted Annotation std'] = annot_data['Weighted Annotation std'][exclude_idx]
            annot_data['Names'] = annot_data['Names'][exclude_idx]

        if keep_annotations is not None:

            include_idx = [c in keep_annotations
                           for c in annot_data['Names']]

            annot_data['Covariance'] = annot_data['Covariance'][include_idx, include_idx]
            annot_data['MAF Normalized sum'] = annot_data['MAF Normalized sum'][include_idx]
            annot_data['Annotation std'] = annot_data['Annotation std'][include_idx]
            annot_data['Weighted Annotation std'] = annot_data['Weighted Annotation std'][include_idx]
            annot_data['Names'] = annot_data['Names'][include_idx]

        # -----------------------------------------
        # Reading LD Scores:
        all_ld_scores = []

        #all_ld_scores += read_baseline_ldscores()
        all_ld_scores += read_modified_ldscores()

        common_snps = list(set.intersection(*map(set, [ldc['LDScores']['SNP'] for ldc in all_ld_scores])))

        # -----------------------------------------
        # Reading sum_stats file:

        gwas_traits = pd.read_csv(sumstats_file)

        args = [
            (all_ld_scores,
            trait,
            annot_data,
            chi2_filter,
            lds_filter) for _, trait in gwas_traits.iterrows()
        ]

        pool = Pool(3)
        res = pool.starmap(perform_ldsc_regression, args)
        pool.close()
        pool.join()

        """
        for _, trait in gwas_traits.iterrows():

            perform_ldsc_regression(
                all_ld_scores,
                trait,
                annot_data,
                chi2_filter,
                lds_filter
            )
        """