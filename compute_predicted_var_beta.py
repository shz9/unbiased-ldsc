import itertools
import numpy as np
import pandas as pd
import os
import glob
from utils import read_pbz2
import datetime
from multiprocessing import Pool


def compute_annot_stats(annot_file_struct, frq_file_struct):

    """
    This function computes statistics for the model annotations,
    including covariance between the annotations, snps per maf bin,
    and sums and MAF-weighted sums of each annotation.
    """

    w_annot_dfs = []

    for chr_num in range(22, 0, -1):

        print(f"Reading Chromosome {chr_num}")

        # Read the annotation and frequency dataframes:
        adf = pd.read_csv(annot_file_struct % chr_num, sep="\s+")
        frq_df = pd.read_csv(frq_file_struct % chr_num, sep="\s+")

        frq_df['MAFVAR'] = 2.*frq_df['MAF']*(1. - frq_df['MAF'])
        adf = adf.loc[adf['SNP'].isin(frq_df['SNP']), ]

        mna_df = pd.merge(frq_df[['SNP', 'MAF', 'MAFVAR']],
                          adf.iloc[:, [2] + list(range(4, len(adf.columns)))])

        w_annot_dfs.append(mna_df)

    w_annots = pd.concat(w_annot_dfs)

    snps_per_bin = {
        b_an: w_annots.loc[w_annots[b_an] == 1, 'SNP'].values for b_an in w_annots.columns
        if 'MAFbin' in b_an
    }

    snps_per_bin['MAFbin0'] = w_annots.loc[
        w_annots[[f'MAFbin{i}' for i in range(1, 11)]].eq(0).all(1),
        'SNP'
    ].values

    annotated_snps = w_annots.loc[
        w_annots[highly_enriched_cats].eq(1).any(1),
        'SNP'
    ].values

    w_annots.set_index('SNP', inplace=True)

    mafvars = w_annots[['MAFVAR']].copy()
    w_annots = w_annots.iloc[:, 2:]

    for k, v in snps_per_bin.items():
        print(k, len(v))

    return {
        'Annotations': w_annots,
        'MAFVAR': mafvars,
        'SNPs per bin': snps_per_bin,
        'Annotated SNPs': annotated_snps
    }


def compute_pred_var_beta(trait_file):

    print(datetime.datetime.now(), trait_file)

    trait_res = read_pbz2(trait_file)

    coef = np.array([list(zip(*trait_res['Coefficients']))[1]])
    #hg2 = trait_res['hg2']

    trait_name = os.path.basename(os.path.dirname(trait_file))
    trait_method = os.path.basename(trait_file).replace('.pbz2', '')

    #ss_df = pd.read_csv(os.path.join("data/independent_sumstats/",
    #                                 trait_name + ".sumstats.gz"), sep="\t")
    #ss_df = ss_df.loc[ss_df['Z']**2 <= max(80., 0.001*np.mean(ss_df['N'])), ]
    #snp_index = annot_data['Annotations'].index.isin(ss_df['SNP'])

    alpha = float(trait_method.split("_")[1])
    mafvar = annot_data['MAFVAR'].values**(-alpha) #[snp_index]**(-alpha)
    mafvar_h2 = annot_data['MAFVAR'].values**(1.-alpha) #[snp_index]**(1.-alpha)
    annots = annot_data['Annotations'].iloc[:, :coef.shape[1]] #[snp_index].iloc[:, :coef.shape[1]]

    m_annots = np.multiply(annots, mafvar)
    m_annots_h2 = np.multiply(annots, mafvar_h2)

    hg2 = np.sum(
        np.dot(m_annots_h2,
               coef.T)
    )

    print(trait_name, trait_method, hg2)

    results = []

    for mbin, mbin_snps in annot_data['SNPs per bin'].items():

        print(datetime.datetime.now(), mbin)

        results.append(
            {
                'Method': trait_method,
                'Trait': trait_name,
                'MAF bin': mbin,
                'Metric': 'Mean Var(beta)',
                'Score': np.mean(
                    np.dot(m_annots.loc[m_annots.index.isin(mbin_snps)],
                           coef.T)
                ),
                'Highly Enriched': False,
                'N': len(m_annots.loc[m_annots.index.isin(mbin_snps)])
            }
        )

        results.append(
            {
                'Method': trait_method,
                'Trait': trait_name,
                'MAF bin': mbin,
                'Metric': 'Heritability',
                'Score': np.sum(
                    np.dot(m_annots_h2.loc[m_annots_h2.index.isin(mbin_snps)],
                           coef.T)
                ),
                'Highly Enriched': False,
                'N': len(m_annots_h2.loc[m_annots_h2.index.isin(mbin_snps)])
            }
        )

        results.append(
            {
                'Method': trait_method,
                'Trait': trait_name,
                'MAF bin': mbin,
                'Metric': 'Prop. Heritability',
                'Score': results[-1]['Score'] / hg2,
                'Highly Enriched': False,
                'N': results[-1]['N']
            }
        )

        results.append(
            {
                'Method': trait_method,
                'Trait': trait_name,
                'MAF bin': mbin,
                'Metric': 'Mean Var(beta)',
                'Score': np.mean(
                    np.dot(m_annots.loc[m_annots.index.isin(mbin_snps) &
                                        m_annots.index.isin(annot_data['Annotated SNPs'])
                           ],
                           coef.T)
                ),
                'Highly Enriched': True,
                'N': len(m_annots.loc[m_annots.index.isin(mbin_snps) &
                                      m_annots.index.isin(annot_data['Annotated SNPs'])
                         ])
            }
        )

        results.append(
            {
                'Method': trait_method,
                'Trait': trait_name,
                'MAF bin': mbin,
                'Metric': 'Heritability',
                'Score': np.sum(
                    np.dot(m_annots_h2.loc[m_annots_h2.index.isin(mbin_snps) &
                                      m_annots_h2.index.isin(annot_data['Annotated SNPs'])],
                           coef.T)
                ),
                'Highly Enriched': True,
                'N': len(m_annots_h2.loc[m_annots_h2.index.isin(mbin_snps) &
                                         m_annots_h2.index.isin(annot_data['Annotated SNPs'])])
            }
        )

        results.append(
            {
                'Method': trait_method,
                'Trait': trait_name,
                'MAF bin': mbin,
                'Metric': 'Prop. Heritability',
                'Score': results[-1]['Score'] / hg2,
                'Highly Enriched': True,
                'N': results[-1]['N']
            }
        )

    print(pd.DataFrame(results))

    return results


if __name__ == '__main__':

    highly_enriched_cats = [
        'Coding_UCSC',
        'Conserved_LindbladToh',
        'GERP.RSsup4',
        'non_synonymous',
        'Conserved_Vertebrate_phastCons46way',
        'Conserved_Mammal_phastCons46way',
        'Conserved_Primate_phastCons46way',
        'BivFlnk',
        'Ancient_Sequence_Age_Human_Promoter',
        'Human_Promoter_Villar_ExAC'
    ]

    annot_data = compute_annot_stats("data/ld_scores/1000G_Phase3_EUR_baselineLD_v2.2_ldscores/baselineLD.%d.annot.gz",
                                     "data/genotype_files/1000G_Phase3_EUR_plinkfiles/1000G.EUR.QC.%d.frq")

    files = [f for f in glob.glob(f"results/regression/EUR/M_5_50_chi2filt/*/*.pbz2")
             if 'D2_' in f]

    pool = Pool(1)
    res = pool.map(compute_pred_var_beta, files[:2])
    pool.close()
    pool.join()

    res_df = pd.DataFrame(list(itertools.chain.from_iterable(res)))
    print(res_df.head())
    res_df.to_csv("mean_pred_var_beta_annot.csv")
