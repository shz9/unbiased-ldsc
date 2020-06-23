from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from utils import makedir, write_pbz2
from collections.abc import Iterable
import os


def evaluate_model_differences(joint_df, maf_df):

    joint_df = pd.merge(joint_df, maf_df[['SNP', 'MAF']], left_on='SNP', right_on='SNP')

    lds_pairs = [
        ['L2', 'LD2'],
        ['L2MAF', 'LD2MAF']
    ]

    results = {}

    for sc1, sc2 in lds_pairs:
        reg_df = joint_df[['SNP', 'CHR', 'MAF', sc1, sc2]]

        slope, intercept, r_value, p_value, std_err = stats.linregress(reg_df[sc1],
                                                                       reg_df[sc2])

        results[f"{sc2} ~ {sc1}"] = {
            'slope': slope,
            'intercept': intercept,
            'R-squared': r_value**2,
            'p-value': p_value
        }

        reg_df['Diff'] = reg_df[sc2] - reg_df[sc1]

        slope, intercept, r_value, p_value, std_err = stats.linregress(reg_df['MAF'],
                                                                       reg_df['Diff'])

        results[f"{sc2} - {sc1} ~ MAF"] = {
            'slope': slope,
            'intercept': intercept,
            'R-squared': r_value ** 2,
            'p-value': p_value
        }

    return results


def plot_manhattan_ld(joint_df, ld_col, title,
                      output_fname, pos_col='BP'):

    starting_pos = 0
    ticks = []
    chrom_spacing = 25000000

    plt.figure(figsize=(24, 10))

    plt.axhline(0.0, ls='--', zorder=1,
                color='#263640')
    plt.axhline(joint_df[ld_col].mean(), ls='--', zorder=1,
                color='#D60A1F', label='Mean')

    unique_chr = sorted(joint_df['CHR'].unique())

    for i, ch in enumerate(unique_chr):

        chr_df = joint_df.loc[joint_df['CHR'] == ch]

        max_pos = chr_df[pos_col].max()

        xmin = starting_pos - (chrom_spacing / 2)
        xmax = max_pos + starting_pos + (chrom_spacing / 2)
        if i % 2 == 1:
            plt.axvspan(xmin=xmin, xmax=xmax, zorder=0, color='#F8F8F8')

        ticks.append((xmin + xmax) / 2)

        plt.scatter(chr_df[pos_col] + starting_pos, chr_df[ld_col],
                    c=chr_df['MAF'], cmap='cool', alpha=0.3, label=None,
                    marker=scatter_mrkr, s=2.0)

        starting_pos += max_pos + chrom_spacing

    plt.xlim([-chrom_spacing, starting_pos])

    plt.xticks(ticks, unique_chr)
    cbar = plt.colorbar()
    cbar.set_label('MAF')

    plt.xlabel("Genomic Position")
    plt.ylabel(ld_score_names[ld_col])

    if include_title:
        plt.title(title)

    plt.legend()

    plt.savefig(output_fname)
    plt.close()


def plot_ld_score_comparison_scatter(joint_df,
                                     compare_against,
                                     title,
                                     output_fname,
                                     base_score="L2",
                                     chr_num=None,
                                     color_col=None):

    if not isinstance(compare_against, Iterable) or type(compare_against) == str:
        compare_against = [compare_against]

    if chr_num is not None:
        joint_df = joint_df.loc[joint_df['CHR'] == chr_num, ]

    plt.figure(figsize=(10, 8))

    for sc in compare_against:

        if len(compare_against) > 1:
            label = ld_score_names[sc]
        else:
            label = None

        if color_col is None:
            plt.scatter(joint_df[base_score], joint_df[sc],
                        color=['skyblue', None][len(compare_against) > 1],
                        marker=scatter_mrkr, alpha=0.3,
                        label=label)
        else:
            plt.scatter(joint_df[base_score], joint_df[sc],
                        c=joint_df[color_col], cmap='cool',
                        marker=scatter_mrkr, alpha=0.3,
                        label=label)

    # Plots line
    x = np.linspace(0.0, joint_df[compare_against + [base_score]].max().max(), 1000)
    plt.plot(x, x, linestyle='--', color='#D60A1F')

    plt.xlabel(ld_score_names[base_score])

    if len(compare_against) > 1:
        plt.ylabel("Unbiased LD Score")
    else:
        plt.ylabel(ld_score_names[compare_against[0]])

    if color_col is not None:
        cbar = plt.colorbar()
        cbar.set_label(color_col)

    if len(compare_against) > 1:
        plt.legend()

    if include_title:
        plt.title(title)

    plt.tight_layout()

    plt.savefig(output_fname)
    plt.close()


def plot_cross_population_comparison(pop1_df, pop1_name,
                                     pop2_df, pop2_name,
                                     ld_col, output_fname):

    joint_df = pd.merge(pop1_df[['SNP', ld_col]],
                        pop2_df[['SNP', ld_col]],
                        on='SNP',
                        suffixes=('_' + pop1_name, '_' + pop2_name))

    plt.figure(figsize=(10, 8))

    sns.scatterplot(x=ld_col + '_' + pop1_name,
                    y=ld_col + '_' + pop2_name,
                    marker=scatter_mrkr,
                    data=joint_df, alpha=0.3, color='skyblue')

    if 'DIFF' in ld_col:
        plt.axhline(y=0.0, color='grey', linestyle='--')
        plt.axvline(x=0.0, color='grey', linestyle='--')
    else:
        x = np.linspace(0.0, joint_df[[ld_col + '_' + pop1_name,
                                       ld_col + '_' + pop2_name]].max().max(), 1000)
        plt.plot(x, x, linestyle='--', color='grey')

    plt.errorbar(joint_df[ld_col + '_' + pop1_name].mean(),
                 joint_df[ld_col + '_' + pop2_name].mean(),
                 xerr=joint_df[ld_col + '_' + pop1_name].std(),
                 yerr=joint_df[ld_col + '_' + pop2_name].std(),
                 ecolor='#D60A1F', capsize=2)

    plt.xlabel(pop1_name)
    plt.ylabel(pop2_name)

    if include_title:
        plt.title(ld_score_names[ld_col])

    plt.tight_layout()

    plt.savefig(output_fname)
    plt.close()


def get_univariate_predictor(lds_df, maf_df, scaled=True):
    """
    The univariate predictor is the term in the univariate
    LD Score regression that doesn't depend on h_g and N.
    e.g. in the Bulik-Sullivan et al. equation, uni_pred = (lj/M).

    For the purposes of numerical stability, it may be desirable
    to work with the scaled univariate predictor M*uni_pred.
    """

    lds_df['LD2'] = lds_df['LD2'] / np.mean(maf_df['VAR'])

    if not scaled:
        for ldn in ['bs_L2'] + ld_scores:
            lds_df[ldn] /= len(maf_df)
    else:
        return lds_df


if __name__ == '__main__':

    plot_dir = "figures/ld_score_comparison/"
    analysis_dir = "results/analysis/ld_score_comparison"

    scatter_mrkr = '.'
    fig_format = '.png'
    sns.set_context('talk')
    include_title = False

    populations = ['EUR', 'EAS', 'AFR']
    ld_scores = ['L2', 'LD2', 'LD2MAF']
    ld_score_diff = ['DIFF_LD2_L2', 'DIFF_LD2MAF_L2MAF', 'DIFF_LD2_LD2MAF']

    ld_score_names = {
        'LDSC': '$r^2 (\\alpha=0)$ LD Score',
        'L2': '$r^2 (\\alpha=0)$ LD Score',
        'L2MAF': '$r^2 (\\alpha=1)$ LD Score',
        'LD2': '$D^2 (\\alpha=0)$ LD Score',
        'LD2MAF': '$D^2 (\\alpha = 1)$ LD Score',
        'DIFF_LD2_L2': 'LD Score Difference $D^2 - r^2$ ($\\alpha = 0$)',
        'DIFF_LD2MAF_L2MAF': 'LD Score Difference $D^2 - r^2$ ($\\alpha = 1$)',
        'DIFF_LD2_LD2MAF': 'LD Score Difference $D^2 (\\alpha = 0) - D^2 (\\alpha = 1)$',
    }

    cols_to_keep = ['CHR', 'CM', 'BP', 'MAF']

    frq_file = "data/genotype_files/1000G_Phase3_%s_plinkfiles/1000G.%s.QC.%d.frq"
    modified_ld_score_dir = "output/ld_scores/1000G_Phase3_%s_mldscores/%s/LD.%d.l2.ldscore.gz"

    pop_scores = {}

    for pop in populations:

        print(f">>> Reading LD Scores for {pop}...")

        chr_dfs = []
        frq_dfs = []

        pop_outdir = os.path.join(plot_dir, pop)
        makedir(pop_outdir)

        analysis_dir = os.path.join(analysis_dir, pop)
        makedir(analysis_dir)

        # ----------------------------------------------------------
        # *** Reading data ***

        for chr_num in range(22, 0, -1):

            print(f">> Reading Chromosome {chr_num}...")

            ld_dfs = []

            for i, lds in enumerate(ld_scores):
                if i == 0:
                    ld_dfs.append(pd.read_csv(modified_ld_score_dir % (pop, lds, chr_num),
                                              sep="\t", index_col=1)[cols_to_keep + ['base' + lds]])
                    ld_dfs[-1]['SNP'] = ld_dfs[-1].index
                else:
                    ld_dfs.append(pd.read_csv(modified_ld_score_dir % (pop, lds, chr_num),
                                              sep="\t", index_col=1)[['base' + lds]])

            chr_df = pd.concat(ld_dfs, axis=1)

            frq_df = pd.read_csv(frq_file % (pop, pop, chr_num), sep="\s+")
            frq_df['VAR'] = 2*frq_df['MAF']*(1. - frq_df['MAF'])

            chr_df.columns = cols_to_keep + ld_scores[:1] + ['SNP'] + ld_scores[1:]

            chr_dfs.append(chr_df)
            frq_dfs.append(frq_df)

        # ----------------------------------------------------------
        # *** Transforming data ***

        all_snps_ld_df = pd.concat(chr_dfs, ignore_index=True)
        all_snps_frq_df = pd.concat(frq_dfs)

        all_snps_univar_pred = get_univariate_predictor(all_snps_ld_df, all_snps_frq_df)

        eval_res = evaluate_model_differences(all_snps_univar_pred, all_snps_frq_df)
        write_pbz2(os.path.join(analysis_dir, "results.pbz2"), eval_res)

        # ----------------------------------------------------------
        # *** LD Score Comparison Plots ***

        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2.png"))
        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2_mafcol.png"),
                                         color_col='MAF')

        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2MAF',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2MAF.png"))
        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2MAF',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2MAF_mafcol.png"),
                                         color_col='MAF')

        plot_ld_score_comparison_scatter(all_snps_univar_pred, ['LD2', 'LD2MAF'],
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2_LD2MAF.png"))

        # ----------------------------------------------------------
        # *** Manhattan plots ***

        for lds in ld_scores:
            plot_manhattan_ld(all_snps_univar_pred, lds,
                              f"LD Scores in 1000G ({pop})",
                              os.path.join(pop_outdir,
                                           lds + f"_manhattan{fig_format}"))

        for lds_diff in ld_score_diff:
            _, ld1, ld2 = lds_diff.split("_")

            all_snps_univar_pred[lds_diff] = all_snps_univar_pred[ld1] - all_snps_univar_pred[ld2]

            plot_manhattan_ld(all_snps_univar_pred, lds_diff,
                              f"LD Scores in 1000G ({pop})",
                              os.path.join(pop_outdir, f"{lds_diff}_manhattan{fig_format}"))

        pop_scores[pop] = all_snps_univar_pred

    print(">>> Generating cross-population plots...")

    for pop1, pop2 in [('EUR', 'AFR'), ('EUR', 'EAS'), ('EAS', 'AFR')]:

        popvs_outdir = os.path.join(plot_dir, pop1 + '_vs_' + pop2)
        makedir(popvs_outdir)

        for lds in ld_scores + ld_score_diff:

            plot_cross_population_comparison(pop_scores[pop1],
                                             pop1,
                                             pop_scores[pop2],
                                             pop2,
                                             lds,
                                             os.path.join(popvs_outdir, lds + fig_format))

