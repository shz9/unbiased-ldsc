from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from utils import makedir, write_pbz2
from collections.abc import Iterable
from multiprocessing import Pool
import os


def evaluate_model_differences(joint_df):

    lds_pairs = [
        ['R2_0.0', 'D2_0.0'],
        ['R2_1.0', 'D2_1.0'],
        ['D2_1.0', 'D2_0.0'],
        ['D2_1.0', 'D2_0.025'],
        ['D2_1.0', 'D2_0.5'],
        ['D2_1.0', 'D2_0.75']
    ]

    results = {}

    for sc1, sc2 in lds_pairs:

        try:
            reg_df = joint_df[['SNP', 'CHR', 'MAF', sc1, sc2]]
        except Exception:
            continue

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
                      output_fname, pos_col='BP', alpha=0.3):

    starting_pos = 0
    ticks = []
    chrom_spacing = 25000000

    plt.figure(figsize=(18, 9))

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
                    c=chr_df['MAF'], cmap='cool', alpha=alpha, label=None,
                    marker=scatter_mrkr, s=2.0)

        starting_pos += max_pos + chrom_spacing

    plt.xlim([-chrom_spacing, starting_pos])

    plt.xticks(ticks, unique_chr)
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label('MAF')

    if 'RELDIFF' in ld_col:
        plt.ylim(-0., 100.)

    plt.xlabel("Genomic Position")
    plt.ylabel(ld_score_names[ld_col])

    if include_title:
        plt.title(title)

    plt.legend()
    plt.tight_layout()

    plt.savefig(output_fname)
    plt.close()


def plot_ld_score_comparison_scatter(joint_df,
                                     compare_against,
                                     title,
                                     output_fname,
                                     base_score="R2_1.0",
                                     chr_num=None,
                                     color_col=None):

    if not isinstance(compare_against, Iterable) or type(compare_against) == str:
        compare_against = [compare_against]

    if chr_num is not None:
        joint_df = joint_df.loc[joint_df['CHR'] == chr_num, ]

    plt.figure(figsize=figure_size)

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

    if 'RELDIFF' in compare_against[0]:
        plt.ylim(-100., 100.)
    else:
        # Plots line
        x = np.linspace(0.0, joint_df[compare_against + [base_score]].max().max(), 1000)
        plt.plot(x, x, linestyle='--', color='#D60A1F')

    plt.xlabel(ld_score_names[base_score])

    if len(compare_against) > 1:
        plt.ylabel("Unbiased LD Score")
    else:
        plt.ylabel(ld_score_names[compare_against[0]])

    if color_col is not None:
        cbar = plt.colorbar(pad=0.05)
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

    plt.figure(figsize=figure_size)

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


def plot_sample_size_comparison(ref_pop='EUR',
                                sample_pop=('EUR20', 'EUR50', 'EUR100', 'EUR200'),
                                comp_ld_scores=('R2_1.0', 'D2_1.0', 'UR2_1.0')):

    res = []

    comp_ld_scores = list(comp_ld_scores)

    for sp in sample_pop:
        joint_df = pd.merge(pop_scores[ref_pop][['SNP'] + comp_ld_scores],
                            pop_scores[sp][['SNP'] + comp_ld_scores],
                            on='SNP',
                            suffixes=('_' + ref_pop, '_' + sp))

        for lds in comp_ld_scores:
            slope, intercept, r_value, p_value, std_err = stats.linregress(joint_df[f"{lds}_{sp}"],
                             joint_df[f"{lds}_{ref_pop}"])

            res.append({
                'Sample Size': int(sp.replace(ref_pop, "")),
                'Estimator': lds,
                'R^2': r_value**2,
                'Relative bias': np.mean((joint_df[f"{lds}_{sp}"] -
                                          joint_df[f"{lds}_{ref_pop}"]) / joint_df[f"{lds}_{ref_pop}"])
            })

    res_data = pd.DataFrame(res)

    res_data.to_csv(os.path.join(analysis_dir, "sample_size_comparison.csv"))

    estimator_labels = ['Corrected $r^2$', 'Unbiased $D^2$', 'Raw $r^2$']

    plt.figure(figsize=figure_size)

    g = sns.lineplot(x="Sample Size", y="R^2",
                     hue="Estimator", style="Estimator",
                     palette=ld_scores_colors, markers=True,
                     dashes=False, data=res_data)
    for t, l in zip(g.legend().texts[1:], estimator_labels):
        t.set_text(l)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sample_size_comparison_r2.png"))

    plt.close()

    plt.figure(figsize=figure_size)

    g = sns.lineplot(x="Sample Size", y="Relative bias",
                     hue="Estimator", style="Estimator",
                     palette=ld_scores_colors, markers=True,
                     dashes=False, data=res_data)
    for t, l in zip(g.legend().texts[1:], estimator_labels):
        t.set_text(l)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sample_size_comparison_relative_bias.png"))

    plt.close()


def get_univariate_predictor(lds_df, maf_df, scaled=True):
    """
    The univariate predictor is the term in the univariate
    LD Score regression that doesn't depend on h_g and N.
    e.g. in the Bulik-Sullivan et al. equation, uni_pred = (lj/M).

    For the purposes of numerical stability, it may be desirable
    to work with the scaled univariate predictor M*uni_pred.
    """

    for ld_col in lds_df.columns:
        if ld_col in ld_scores:
            alpha = float(ld_col.split("_")[-1])

            lds_df[ld_col] /= np.mean(maf_df['VAR']**(1. - alpha))

    if not scaled:
        for ldn in ld_scores:
            lds_df[ldn] /= len(maf_df)
    else:
        return lds_df


def read_merge_ld_scores(chr_num):

    ld_dfs = []
    processed_ld_scores = []

    for i, lds in enumerate(ld_scores):
        try:

            if i == 0:
                ld_dfs.append(pd.read_csv(modified_ld_score_dir % (pop, lds, chr_num),
                                          sep="\t", index_col=1)[cols_to_keep + ['base' + lds]])
                ld_dfs[-1]['SNP'] = ld_dfs[-1].index
            else:
                ld_dfs.append(pd.read_csv(modified_ld_score_dir % (pop, lds, chr_num),
                                          sep="\t", index_col=1)[['base' + lds]])

            processed_ld_scores.append(lds)

        except Exception as e:
            print(e)
            continue

    chr_df = pd.concat(ld_dfs, axis=1)

    frq_df = pd.read_csv(frq_file % (pop, pop, chr_num), sep="\s+")
    frq_df['VAR'] = 2*frq_df['MAF']*(1. - frq_df['MAF'])

    chr_df.columns = cols_to_keep + processed_ld_scores[:1] + ['SNP'] + processed_ld_scores[1:]

    return chr_df, frq_df


if __name__ == '__main__':

    plot_dir = "figures/ld_score_comparison_test/"
    analysis_dir = "results/analysis/ld_score_comparison_test"

    num_procs = 6

    figure_size = (7.5, 6)
    scatter_mrkr = '.'
    fig_format = '.png'
    sns.set_context('talk')
    include_title = False

    populations = ['EUR', 'EUR20', 'EUR50', 'EUR100', 'EUR200', 'EAS', 'AFR']

    ld_scores = ['R2_0.0', 'R2_0.25', 'R2_0.5', 'R2_0.75', 'R2_1.0',
                 'D2_0.0', 'D2_0.25', 'D2_0.5', 'D2_0.75', 'D2_1.0',
                 'UR2_0.0', 'UR2_1.0']

    ld_score_diff = ['DIFF_D2_1.0_R2_1.0'] #['DIFF_LD2_L2', 'DIFF_LD2MAF_L2MAF', 'DIFF_LD2_LD2MAF']
    rel_ld_score_diff = ['RELDIFF_D2_1.0_R2_1.0']  # ['RELDIFF_LD2_L2', 'RELDIFF_LD2MAF_L2MAF', 'RELDIFF_LD2_LD2MAF']

    ld_scores_colors = {
        'D2_0.0': '#E15759',
        'D2_1.0': '#76B7B2',
        'R2_0.0': '#B07AA1',
        'R2_1.0': '#F28E2B',
        'UR2_0.0': '#1f77b4',
        'UR2_1.0': '#1f77b4'
    }

    ld_score_names = {
        'R2_0.0': '$r^2 (\\alpha=0)$ LD Score',
        'R2_0.25': '$r^2 (\\alpha=0.25)$ LD Score',
        'R2_0.5': '$r^2 (\\alpha=0.5)$ LD Score',
        'R2_0.75': '$r^2 (\\alpha=0.75)$ LD Score',
        'R2_1.0': '$r^2 (\\alpha=1)$ LD Score',
        'D2_0.0': '$D^2 (\\alpha=0)$ LD Score',
        'D2_0.25': '$D^2 (\\alpha=0.25)$ LD Score',
        'D2_0.5': '$D^2 (\\alpha=0.5)$ LD Score',
        'D2_0.75': '$D^2 (\\alpha=0.75)$ LD Score',
        'D2_1.0': '$D^2 (\\alpha = 1)$ LD Score',
        'DIFF_LD2_L2': 'LD Score Difference $D^2 - r^2$ ($\\alpha = 0$)',
        'DIFF_LD2MAF_L2MAF': 'LD Score Difference $D^2 - r^2$ ($\\alpha = 1$)',
        'DIFF_LD2_LD2MAF': 'LD Score Difference $D^2 (\\alpha = 0) - D^2 (\\alpha = 1)$',
        'RELDIFF_LD2_L2': '% Difference in LD Scores ($\\alpha = 0$)\n$abs(D^2 - r^2) / r^2$',
        'RELDIFF_LD2MAF_L2MAF': '% Absolute Difference in LD Scores ($\\alpha = 1$)\n$abs(D^2 - r^2) / r^2$',
        'RELDIFF_LD2_LD2MAF': '% Absolute Difference in LD Scores\n$(D^2 (\\alpha = 0) - D^2 (\\alpha = 1)) / (D^2 (\\alpha = 1))$',
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

        pop_analysis_dir = os.path.join(analysis_dir, pop)
        makedir(pop_analysis_dir)

        # ----------------------------------------------------------
        # *** Reading data ***

        pool = Pool(num_procs)
        res = pool.map(read_merge_ld_scores, list(range(22, 0, -1)))
        pool.close()
        pool.join()

        chr_dfs = [r[0] for r in res]
        frq_dfs = [r[1] for r in res]

        # ----------------------------------------------------------
        # *** Transforming data ***

        all_snps_ld_df = pd.concat(chr_dfs, ignore_index=True)
        all_snps_frq_df = pd.concat(frq_dfs)

        all_snps_univar_pred = get_univariate_predictor(all_snps_ld_df, all_snps_frq_df)

        eval_res = evaluate_model_differences(all_snps_univar_pred)
        write_pbz2(os.path.join(pop_analysis_dir, "results.pbz2"), eval_res)

        pop_scores[pop] = all_snps_univar_pred

        if pop in ['EUR20', 'EUR50', 'EUR100', 'EUR200']:
            continue

        # ----------------------------------------------------------
        # *** LD Score Comparison Plots ***

        print(">> Generating scatter plots...")

        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'D2_1.0',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_R2_1.0_vs_D2_1.0_mafcol.png"),
                                         base_score='R2_1.0',
                                         color_col='MAF')

        for y_lds in ['D2_0.0', 'D2_0.25', 'D2_0.5', 'D2_0.75']:
            try:
                plot_ld_score_comparison_scatter(all_snps_univar_pred, y_lds,
                                                 "LD Scores in 1000G (%s)" % pop,
                                                 os.path.join(pop_outdir, f"scatter_D2_1.0_vs_{y_lds}_mafcol.png"),
                                                 base_score='D2_1.0',
                                                 color_col='MAF')
            except KeyError:
                continue

        """
        print(">> Generating scatter plots...")

        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2MAF',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2MAF_vs_LD2MAF_mafcol.png"),
                                         base_score='L2MAF',
                                         color_col='MAF')

        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_LD2MAF_vs_LD2_mafcol.png"),
                                         base_score='LD2MAF',
                                         color_col='MAF')
        
        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2_mafcol.png"),
                                         color_col='MAF')
                                         
        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2_vs_LD2.png"))
                                         
        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2MAF',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_L2MAF_vs_LD2MAF.png"),
                                         base_score='L2MAF')
        plot_ld_score_comparison_scatter(all_snps_univar_pred, 'LD2',
                                         "LD Scores in 1000G (%s)" % pop,
                                         os.path.join(pop_outdir, "scatter_LD2MAF_vs_LD2.png"),
                                         base_score='LD2MAF')
        """

        # ----------------------------------------------------------
        # *** Manhattan plots ***

        print(">> Generating Manhattan plots...")

        """
        for lds in ld_scores:
            if lds not in ['UL2', 'UL2MAF']:
                plot_manhattan_ld(all_snps_univar_pred, lds,
                                  f"LD Scores in 1000G ({pop})",
                                  os.path.join(pop_outdir,
                                               f"manhattan_{lds}{fig_format}"))

        for lds_diff in ld_score_diff:
            print(lds_diff)
            _, ld1, ld2 = lds_diff.split("_")

            all_snps_univar_pred[lds_diff] = all_snps_univar_pred[ld1] - all_snps_univar_pred[ld2]

            #if lds_diff == 'DIFF_LD2MAF_L2MAF':
            #    all_snps_univar_pred.to_csv(f"./cache/{lds_diff}.csv")

            plot_manhattan_ld(all_snps_univar_pred, lds_diff,
                              f"LD Scores in 1000G ({pop})",
                              os.path.join(pop_outdir, f"manhattan_{lds_diff}{fig_format}"))
                              

        for lds_diff in rel_ld_score_diff:
            print(lds_diff)
            _, ld1, ld2 = lds_diff.split("_")
            all_snps_univar_pred[lds_diff] = 100.*np.abs(all_snps_univar_pred[ld1] -
                                                         all_snps_univar_pred[ld2]) / all_snps_univar_pred[ld2]

            plot_manhattan_ld(all_snps_univar_pred, lds_diff,
                              f"LD Scores in 1000G ({pop})",
                              os.path.join(pop_outdir, f"manhattan_{lds_diff}{fig_format}"),
                              alpha=0.8)

            plot_ld_score_comparison_scatter(all_snps_univar_pred, lds_diff,
                                             "LD Scores in 1000G (%s)" % pop,
                                             os.path.join(pop_outdir, f"scatter_L2MAF_vs_{lds_diff}_mafcol.png"),
                                             base_score=ld2,
                                             color_col='MAF')
        """

        pop_scores[pop] = all_snps_univar_pred

    """
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
    """

    print(">>> Generating sample size plots...")
    plot_sample_size_comparison()

