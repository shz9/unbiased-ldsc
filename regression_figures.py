from collections.abc import Iterable
import itertools
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import glob
import os
import sys
from utils import read_pbz2, makedir, get_multiplicative_factor
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

plt.rcParams['hatch.linewidth'] = 2
r_meta = importr('meta')


def meta_analyze_annot(annot_df, metric='tau_star', ma_type='random'):
    meta_results = []
    meta_results_se = []

    for method in annot_df['Method'].unique():

        m_annot_df = annot_df.loc[(annot_df['Method'] == method) &
                                  (annot_df['Metric'] == metric)]

        piv_annot = m_annot_df.pivot(index='Annotation', columns='Trait', values='Score')
        piv_annot_se = m_annot_df.pivot(index='Annotation', columns='Trait', values='Score SE')

        meta_res = {'Method': method}
        meta_res_se = {'Method': method}

        for annot in list(piv_annot.index):

            res = dict(r_meta.metagen(ro.FloatVector(piv_annot.loc[annot].values),
                                      ro.FloatVector(piv_annot_se.loc[annot].values)).items())

            if ma_type == 'fixed':
                meta_res[annot] = list(res['TE.fixed'].items())[0][1]
                meta_res_se[annot] = list(res['seTE.fixed'].items())[0][1]
            elif ma_type == 'random':
                meta_res[annot] = list(res['TE.random'].items())[0][1]
                meta_res_se[annot] = list(res['seTE.random'].items())[0][1]

        meta_results.append(meta_res)
        meta_results_se.append(meta_res_se)

    return (pd.DataFrame(meta_results).set_index('Method').T,
            pd.DataFrame(meta_results_se).set_index('Method').T)


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height / 2),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)


def plot_regression_result(data,
                           title,
                           output_fname,
                           chi2_colname='CHISQ'):
    """
    This function produces figures similar to Figure 2. in Bulik-Sullivan et al. (2015)
    :param data: a dictionary or list of dictionary that must have the following keys defined:
    `dataframe`: The regression dataframe
    `Intercept`: The inferred value of the intercept
    `hg2`: The inferred heritability estimate
    `N`: GWAS sample size
    `MC`: Normalization factor (e.g. `M` or `M_5_50` in LDSC).
    `method`: Method name
    :return:
    """

    if isinstance(data, dict):
        data = [data]

    plt.figure(figsize=(10, 8))

    for dt in data:

        gdf = dt['binned_dataframe']

        if len(data) > 1:
            sns.scatterplot(x='binLD', y=chi2_colname, data=gdf, label=f"{dt['method']} data",
                            palette='ch:.25', zorder=10)
        else:
            sns.scatterplot(x='binLD', y=chi2_colname, data=gdf, hue='wLD',
                            palette='ch:.25', zorder=10)

        x = np.linspace(0, max(gdf['binLD']), 1000)

        reg_label = f"{dt['method']} Regression line: ($h_g^2=${dt['hg2']:.3f}, $b_0=${dt['Intercept']:.3f})"
        plt.plot(x, dt['N']*dt['hg2']*x/dt['Annotation Sum'][0][0] + dt['Intercept'], label=reg_label)

        plt.xlabel("LD Score bin")
        plt.ylabel("Mean $\chi^2$")
        plt.title(title)

        plt.xlim(0., np.percentile(gdf['binLD'], 95) + 0.1 * np.percentile(gdf['binLD'], 95))
        plt.ylim(min(gdf[chi2_colname]) - .1 * min(gdf[chi2_colname]),
                 max(gdf[chi2_colname]) + .1 * max(gdf[chi2_colname]))

        leg = plt.legend()

        if len(data) == 1:
            for i, tex in enumerate(leg.get_texts()):
                if tex.get_text() == "wLD":
                    tex.set_text("Regression weight")
                try:
                    tex.set_text(str(round(float(tex.get_text()), 2)))
                except Exception as e:
                    continue

    makedir(os.path.dirname(output_fname))
    plt.savefig(output_fname)
    plt.close()


def plot_global_metric_estimates(trait_name, regression_results,
                                  metric='hg2', default_color='skyblue'):

    methods = ld_scores_ord
    stratified_mean, stratified_yerr, univariate_mean, univariate_yerr = [], [], [], []

    for m in methods:
        stratified_mean.append(regression_results['S-' + m][metric])

        try:
            stratified_yerr.append(regression_results['S-' + m][metric + '_se'])
        except KeyError:
            stratified_yerr.append(None)

        univariate_mean.append(regression_results[m][metric])

        try:
            univariate_yerr.append(regression_results[m][metric + '_se'])
        except KeyError:
            univariate_yerr.append(None)

    x = np.arange(len(methods))
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 7))

    if any([y is None for y in univariate_yerr]):
        rects1 = ax.bar(x - width / 2, univariate_mean, width,
                        color=default_color, label='Univariate')
    else:
        rects1 = ax.bar(x - width / 2, univariate_mean, width,
                        yerr=univariate_yerr, color=default_color, label='Univariate',
                        error_kw=dict(lw=1, capsize=2, capthick=1))

    if any([y is None for y in stratified_yerr]):
        rects2 = ax.bar(x + width / 2, stratified_mean, width, color=default_color,
                        label='Stratified')
    else:
        rects2 = ax.bar(x + width / 2, stratified_mean, width, yerr=stratified_yerr,
                        color=default_color, label='Stratified',
                        error_kw=dict(lw=1, capsize=2, capthick=1))

    for bar in rects1:
        bar.set_hatch('////')
        bar.set_edgecolor('white')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Estimated ' + metric)
    ax.set_title('Estimated ' + metric + ' (' + trait_name + ')')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    fig.tight_layout()

    output_dir = os.path.join(plot_dir, trait_name, 'global_metrics')
    makedir(output_dir)

    plt.savefig(os.path.join(output_dir, metric + fig_format))
    plt.close()


def plot_annotation_estimates(trait_name, regression_results, metric,
                              annot_key='Annotations',
                              print_labels=False):

    methods = ['S-' + ldc for ldc in ld_scores_ord if ldc in regression_results.keys()]
    annotations = np.array(regression_results[methods[0]][annot_key]['Names'])

    keep_index = None
    annotations = annotations[keep_index].ravel()

    sorted_idx = np.argsort(np.abs(regression_results['S-LDSC']
                                   [annot_key][metric].ravel()[keep_index].ravel()))[::-1]
    annotations = annotations[sorted_idx]

    mult_factor = get_multiplicative_factor(
        np.mean([regression_results[m][annot_key][metric][keep_index].ravel()
                 for m in methods])
    )

    bar_vals = {}

    for m in methods:
        try:
            bar_vals[m] = {
                'mean': regression_results[m][annot_key][metric][keep_index].ravel()[sorted_idx] / mult_factor,
                'se': regression_results[m][annot_key][metric + "_se"][keep_index].ravel()[sorted_idx] / mult_factor
            }
        except KeyError:
            bar_vals[m] = {
                'mean': regression_results[m][annot_key][metric].ravel()[keep_index].ravel()[sorted_idx] / mult_factor
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
                                    width, yerr=bar_vals[m]['se'][10*xidx:10*(xidx + 1)],
                                    color=ld_scores_colors[m],
                                    label=method_names[m],
                                    error_kw=dict(lw=1, capsize=2, capthick=1)))
            else:
                rects.append(ax.bar(x - (2 - idx)*width,
                                    bar_vals[m]['mean'][10*xidx:10*(xidx + 1)], width,
                                    color=ld_scores_colors[m],
                                    label=method_names[m]))

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xticks(x)
        ax.set_xticklabels(annotations[10*xidx:10*(xidx + 1)])

        if mult_factor != 1:
            ax.set_ylabel(f'Estimated {metric} (x {mult_factor})')
        else:
            ax.set_ylabel(f'Estimated {metric}')

        if xidx == 0:
            ax.set_title(f'Estimated {metric} of functional annotations ({trait_name})')

        if print_labels:
            [autolabel(rect, ax) for rect in rects]

        for label in ax.get_xmajorticklabels():
            label.set_rotation(90)

    plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

    fig.tight_layout()

    output_dir = os.path.join(plot_dir, trait_name, 'annotations')
    makedir(output_dir)

    plt.savefig(os.path.join(output_dir, metric + fig_format))
    plt.close()


def plot_combined_global_results(metric_df, metric,
                                 relative_to=None,
                                 adjust_intercept=True,
                                 write_table=True,
                                 methods_included='all'):

    metric_df = metric_df.loc[metric_df['Metric'] == metric]

    mean_pivot = metric_df.pivot(index='Trait', columns='Method',
                                 values='Score')
    se_pivot = metric_df.pivot(index='Trait', columns='Method',
                               values='Score SE')

    if relative_to is not None:
        se_pivot = 100.0 * se_pivot.div(mean_pivot[relative_to], axis=0)
        mean_pivot = 100.0 * (mean_pivot.div(mean_pivot[relative_to], axis=0) - 1.0)

        mean_pivot.drop(relative_to, axis=1, inplace=True)
        se_pivot.drop(relative_to, axis=1, inplace=True)
    elif adjust_intercept and metric == 'Intercept':
        mean_pivot -= 1.0

    labels_dict = method_names.copy()
    method_colors = ld_scores_colors.copy()

    if methods_included == 'all':
        methods = [lds for lds in ld_scores_ord
                   if lds in mean_pivot.columns]
        methods += ['S-' + lds for lds in ld_scores_ord
                    if 'S-' + lds in mean_pivot.columns]
    elif methods_included == 'stratified':
        methods = ['S-' + lds for lds in ld_scores_ord
                   if 'S-' + lds in mean_pivot.columns]
        for m in methods:
            labels_dict[m] = labels_dict[m].replace(', \\theta=1', '')

        if relative_to is not None:
            labels_dict[relative_to] = labels_dict[relative_to].replace(', \\theta=1', '')

    elif methods_included == 'figure2':

        methods = [m for m in ['D2_1.0', 'D2_0.0', 'S-D2_1.0', 'S-D2_0.0']
                   if m in mean_pivot.columns]

        for m in methods:
            labels_dict[m] = labels_dict[m].replace('r^2 ', '').replace('D^2 ', '')

        #if 'LD2' in methods:
        #     method_colors['LD2'] = '#76B7B2'
        #if 'S-L2MAF' in methods:
        #    method_colors['S-L2MAF'] = '#B07AA1'

        if relative_to is not None:
            labels_dict[relative_to] = labels_dict[relative_to].replace('r^2 ', '').replace('D^2 ', '')

    else:
        methods = [lds for lds in ld_scores_ord
                   if lds in mean_pivot.columns]
        for m in methods:
            labels_dict[m] = labels_dict[m].replace(', \\theta=0', '')

        if relative_to is not None:
            labels_dict[relative_to] = labels_dict[relative_to].replace(', \\theta=0', '')

    mean_dict = {
        k: [v[t] for t in trait_subset]
        for k, v in mean_pivot.to_dict().items()
    }
    se_dict = {
        k: [v[t] for t in trait_subset]
        for k, v in se_pivot.to_dict().items()
    }

    fig, ax = plt.subplots(figsize=(15, 10))

    width = 1.0
    x = np.arange(len(mean_dict[methods[0]])) * 2 * width * (1 + len(methods) // 2)

    for idx, m in enumerate(methods):
        rect = ax.bar(x - (len(methods) // 2 - idx) * width,
                      mean_dict[m],
                      width, yerr=se_dict[m],
                      label=labels_dict[m],
                      color=method_colors[m],
                      error_kw=dict(lw=1, capsize=2, capthick=1))
        if m[:2] != 'S-':
            for bar in rect:
                bar.set_hatch('////')
                bar.set_edgecolor('white')

    ax.set_xticks(x)
    ax.set_xticklabels(trait_subset)

    for label in ax.get_xmajorticklabels():
        label.set_rotation(90)

    if relative_to is not None:
        plt.ylabel(f"{metric_tex[metric]} (% difference w.r.t. {labels_dict[relative_to]})")
    elif adjust_intercept and metric == 'Intercept':
        plt.ylabel(f"Estimated Confounding (Intercept - 1.0)")
    else:
        plt.ylabel(f"Estimated {metric_tex[metric]}")

    if include_title:
        plt.title(f"{metric_tex[metric]}")
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")

    if relative_to is None:
        if metric == "Intercept" and adjust_intercept:
            relative_to = 'value_adj'
        else:
            relative_to = 'value'
    else:
        relative_to = 'wrt_' + relative_to

    if write_table and methods_included == 'all':
        gm_output_dir = os.path.join(table_dir, "global_metrics")
        makedir(gm_output_dir)

        mean_pivot.applymap(metric_format[metric].format).to_excel(os.path.join(gm_output_dir,
                                                                                f"mean_{metric}_{relative_to}.xls"))

        (mean_pivot.applymap(metric_format[metric].format) + ' (' +
         se_pivot.applymap(metric_format[metric].format) + ')').to_excel(os.path.join(gm_output_dir,
                                                                                      f"{metric}_{relative_to}.xls"))

    output_fname = os.path.join(plot_dir, 'meta', 'global', methods_included,
                                f"{metric}_{relative_to}{fig_format}")
    makedir(os.path.dirname(output_fname))

    plt.tight_layout()
    plt.savefig(output_fname)
    plt.close()


def get_significant_factors(mean_pivot, se_pivot, m1, m2):

    pval = 2. * stats.norm.sf(abs((mean_pivot[m2] - mean_pivot[m1]) / se_pivot[m2]))
    return set(mean_pivot[pval < 0.05].index)


def get_z_score_of_difference(mean_pivot, se_pivot, m1, m2):
    return (mean_pivot[m2] - mean_pivot[m1]) / se_pivot[m2]


def plot_meta_analyzed_annotation_coefficients_scatter(annotation_df,
                                                       metric='tau_star_w',
                                                       x='S-R2_1.0',
                                                       y='S-D2_0.0',
                                                       write_table=True,
                                                       write_pvalue=True):

    mean_piv, se_piv = meta_analyze_annot(annotation_df, metric)

    if metric in ['enrichment', 'enrichment2']:
        drop = enrichment_exclude
    else:
        drop = ['base']

    drop += [an for an in mean_piv.index if 'flanking.500' in an]

    se_piv = se_piv.drop(drop)
    mean_piv = mean_piv.drop(drop)

    if write_table:

        filt_cond = [not (an == 'base' or
                          'MAFbin' in an or
                          'flanking.500' in an) for an in mean_piv.index]

        wt_mean_piv = mean_piv.loc[filt_cond, :]
        wt_se_piv = se_piv.loc[filt_cond, :]

        an_output_dir = os.path.join(table_dir, "meta_analyzed_annotation_metrics")
        makedir(an_output_dir)
        (wt_mean_piv.applymap(metric_format[metric].format) + ' (' +
         wt_se_piv.applymap(metric_format[metric].format) + ')').to_excel(os.path.join(an_output_dir,
                                                                                       f"{metric}.xls"))

        wt_mean_piv.applymap(metric_format[metric].format).to_excel(os.path.join(an_output_dir,
                                                                                 f"mean_{metric}.xls"))

        if write_pvalue:
            pval = 2. * stats.norm.sf(abs(wt_mean_piv / wt_se_piv))
            pval = -np.log10(pval)

            pd.DataFrame(pval, columns=wt_mean_piv.columns, index=wt_mean_piv.index).to_excel(
                os.path.join(an_output_dir, f"{metric}_pvalue.xls")
            )

        wt_se_piv = 100.0 * wt_se_piv.div(wt_mean_piv[x], axis=0)
        wt_mean_piv = 100.0 * (wt_mean_piv.div(wt_mean_piv[x], axis=0) - 1.0)

        wt_mean_piv.drop(x, axis=1, inplace=True)
        wt_se_piv.drop(x, axis=1, inplace=True)

        (wt_mean_piv.applymap(metric_format[metric].format) + ' (' +
         wt_se_piv.applymap(metric_format[metric].format) + ')').to_excel(os.path.join(an_output_dir,
                                                                                       f"{metric}_wrt_S-L2MAF.xls"))

        wt_mean_piv.applymap(metric_format[metric].format).to_excel(os.path.join(an_output_dir,
                                                                                 f"mean_{metric}_wrt_S-L2MAF.xls"))

    sig_annots = get_significant_factors(mean_piv, se_piv, x, y)
    non_sig_annots = set(mean_piv.index) - set(sig_annots)

    mafbin_annots = [f'MAFbin{i}' for i in range(1, 11)]
    ld_annots = ['Backgrd_Selection_Stat', 'CpG_Content_50kb', 'MAF_Adj_ASMC',
                 'MAF_Adj_LLD_AFR', 'Nucleotide_Diversity_10kb', 'Recomb_Rate_10kb']
    func_annots = list(set(mean_piv.index) - set(mafbin_annots).union(set(ld_annots)))

    sig_mafbins = list(set(mafbin_annots).intersection(sig_annots))
    sig_ld_annots = list(set(ld_annots).intersection(sig_annots))
    sig_func_annots = list(set(func_annots).intersection(sig_annots))

    color = mean_piv.index.to_series().apply(lambda s: ['#CFCFCF70', 'red'][s in sig_annots])
    color[sig_mafbins] = '#666666'
    color[sig_ld_annots] = '#e7298a'
    color[sig_func_annots] = '#a6761d'

    plot_cats = {
        'Non-significant Annotations': non_sig_annots,
        'MAF Bins': sig_mafbins,
        'LD-related Annotations': sig_ld_annots,
        'Functional Annotations': sig_func_annots
    }

    plt.subplots(figsize=(10, 8))

    for label, annots in plot_cats.items():
        if len(annots) < 1:
            continue
        if label == 'Non-significant Annotations':
            label = None
            zorder = 1
        else:
            zorder = 3

        plt.errorbar(x=mean_piv.loc[annots, x], y=mean_piv.loc[annots, y],
                     xerr=se_piv.loc[annots, x], yerr=se_piv.loc[annots, y],
                     ecolor=color[annots], ls='none', label=label, zorder=zorder)

    li = np.linspace(mean_piv[[x, y]].values.min(),
                     mean_piv[[x, y]].values.max(),
                     1000)
    plt.plot(li, li, ls='--', zorder=2)

    plt.legend(title="Category of Significantly\nDivergent Estimates")
    # loc='center left', bbox_to_anchor=(1, 0.5)
    plt.xlabel(f"{metric_tex[metric]} - {method_names[x]}")
    plt.ylabel(f"{metric_tex[metric]} - {method_names[y]}")

    plt.tight_layout()

    output_fname = os.path.join(plot_dir, 'meta', 'annotations', 'scatter',
                                f"{metric}_{x}_{y}{fig_format}")
    makedir(os.path.dirname(output_fname))

    plt.savefig(output_fname)
    plt.close()


def plot_meta_analyzed_annotation_coefficients_bar(annotation_df,
                                                   metric='tau_star',
                                                   relative_to=None):

    annot_subset = [
        'MAF_Adj_Predicted_Allele_Age', 'MAF_Adj_LLD_AFR',
        'Recomb_Rate_10kb', 'Nucleotide_Diversity_10kb',
        'Backgrd_Selection_Stat', 'CpG_Content_50kb', 'MAF_Adj_ASMC'
    ]

    annotation_df = annotation_df.loc[(annotation_df['Metric'] == metric) &
                                      (annotation_df['Annotation'].apply(
                                          lambda x: not (x == 'base' or
                                                         'MAFbin' in x or
                                                         'flanking.500' in x)))]

    mean_piv, se_piv = meta_analyze_annot(annotation_df, metric)

    if relative_to is not None:
        se_pivot = 100.0 * se_piv.div(mean_piv[relative_to], axis=0)
        mean_pivot = 100.0 * (mean_piv.div(mean_piv[relative_to], axis=0) - 1.0)

        mean_pivot.drop(relative_to, axis=1, inplace=True)
        se_pivot.drop(relative_to, axis=1, inplace=True)
        mult_factor = 1

    else:

        mult_factor = get_multiplicative_factor(mean_piv.values.mean())
        mean_piv /= mult_factor
        se_piv /= mult_factor

    methods = ['S-' + lds for lds in ld_scores_ord
               if 'S-' + lds in mean_piv.columns]

    mean_dict = {
        k: [v[t] for t in annot_subset]
        for k, v in mean_piv.to_dict().items()
    }
    se_dict = {
        k: [v[t] for t in annot_subset]
        for k, v in se_piv.to_dict().items()
    }

    fig, ax = plt.subplots(figsize=(15, 8))

    width = 1.0
    x = np.arange(len(mean_dict[methods[0]])) * 2 * width * (1 + len(methods) // 2)

    for idx, m in enumerate(methods):
        rect = ax.bar(x - (len(methods) // 2 - idx) * width,
                      mean_dict[m],
                      width, yerr=se_dict[m],
                      label=method_names[m],
                      color=ld_scores_colors[m],
                      error_kw=dict(lw=1, capsize=2, capthick=1))
        if m[:2] != 'S-':
            for bar in rect:
                bar.set_hatch('////')
                bar.set_edgecolor('white')

    ax.set_xticks(x)
    ax.set_xticklabels(annot_subset)

    for label in ax.get_xmajorticklabels():
        label.set_rotation(90)

    if relative_to is None:
        if mult_factor != 1:
            plt.ylabel(f'Meta-analyzed {metric_tex[metric]} (x {mult_factor})')
        else:
            plt.ylabel(f'Meta-analyzed {metric_tex[metric]}')
    else:
        plt.ylabel(f"Meta-analyzed {metric_tex[metric]} (% difference w.r.t. {relative_to})")

    if include_title:
        plt.title(f"Meta-analyzed {metric_tex[metric]}")

    plt.legend(frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")

    if relative_to is None:
        relative_to = 'value'
    else:
        relative_to = 'wrt_' + relative_to

    plt.tight_layout()

    output_fname = os.path.join(plot_dir, 'meta', 'annotations', 'bar',
                                f"{metric}_{relative_to}{fig_format}")
    makedir(os.path.dirname(output_fname))

    plt.savefig(output_fname)
    plt.close()


def plot_trait_vs_annotation_heatmap(annotation_df, metric,
                                     bottom_m='S-R2_1.0', right_m='S-R2_1.0',
                                     left_m='S-D2_0.0', top_m='S-D2_0.0',
                                     write_table=True,
                                     turn_binary=True,
                                     bonf_pval=-np.log10(0.05/53)):

    def quatromatrix(left, bottom, right, top, ax=None, triplotkw={}, tripcolorkw={}):
        """
        From: https://stackoverflow.com/a/44293388
        """

        if not ax: ax = plt.gca()
        n = left.shape[0]
        m = left.shape[1]

        a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
        tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])

        A = np.zeros((n * m * 5, 2))
        Tr = np.zeros((n * m * 4, 3))

        for i in range(n):
            for j in range(m):
                k = i * m + j
                A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
                Tr[k * 4:(k + 1) * 4, :] = tr + k * 5

        C = np.c_[left.flatten(), bottom.flatten(),
                  right.flatten(), top.flatten()].flatten()

        ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
        return ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)

    fdf = annotation_df.loc[(annotation_df['Metric'] == metric) &
                            (annotation_df['Annotation'].apply(lambda x: not (x == 'base' or
                                                                              'MAFbin' in x or
                                                                              'flanking.500' in x)))]

    if metric == 'tau_pvalue' and turn_binary:
        fdf['Score'] = (fdf['Score'] > bonf_pval).astype(int)

        nsig_by_annot = fdf.groupby('Annotation')['Score'].sum()
        include_annot = list(nsig_by_annot[nsig_by_annot > 0].index)
        nsig_by_trait = fdf.groupby('Trait')['Score'].sum()
        include_trait = list(nsig_by_trait[nsig_by_trait > 0].index)

        fdf = fdf.loc[fdf['Trait'].isin(include_trait) & fdf['Annotation'].isin(include_annot)]
    elif metric in ['enrichment', 'enrichment2', 'differential_enrichment']:
        fdf = fdf.loc[~fdf['Annotation'].isin(enrichment_exclude)]

    piv_dict = {}
    piv_se_dict = {}

    s_methods = np.unique([bottom_m, right_m, left_m, top_m])

    for m in s_methods:
        piv_dict[m] = fdf.loc[(fdf['Method'] == m)].pivot(
            index='Annotation', columns='Trait', values='Score'
        )
        piv_se_dict[m] = fdf.loc[(fdf['Method'] == m)].pivot(
            index='Annotation', columns='Trait', values='Score SE'
        )

    annotations = list(piv_dict[s_methods[0]].index)
    traits = list(piv_dict[s_methods[0]].columns)

    if write_table:

        pivots = [piv_dict, piv_se_dict]
        res = []

        for pv in pivots:

            combined_df = pd.concat(
                [df.T for df in pv.values()],
                axis=1, join='inner'
            )
            combined_df = pd.DataFrame(
                combined_df.values, index=combined_df.index,
                columns=pd.MultiIndex.from_tuples(
                    list(zip(combined_df.columns,
                             np.repeat(np.array(s_methods),
                                       len(combined_df.columns) // len(s_methods)))),
                    names=['Annotation', 'Method']
                )
            )

            combined_df = combined_df[[(an, m) for an in pv[s_methods[0]].index for m in s_methods]]
            res.append(combined_df)

        combined_df = res[0].applymap(metric_format[metric].format)

        if metric != 'tau_pvalue':
            combined_df += ' (' + res[1].applymap(metric_format[metric].format) + ')'

        an_output_dir = os.path.join(table_dir, "per_trait_annotation_metrics")
        makedir(an_output_dir)
        combined_df.to_excel(os.path.join(an_output_dir,
                                          f"{metric}{['', '_binary'][metric == 'tau_pvalue' and turn_binary]}.xls"))

    right = piv_dict[right_m].values
    bottom = piv_dict[bottom_m].values
    left = piv_dict[left_m].values
    top = piv_dict[top_m].values

    fig, ax = plt.subplots(figsize=(20, 20))

    if 'tau_star' in metric:
        cmap = plt.cm.get_cmap('coolwarm', 20)
    else:
        cmap = plt.cm.get_cmap('Blues', 20)

    r = quatromatrix(left, bottom, right, top, ax=ax,
                     triplotkw={"color": "#F1F1F1", "lw": 0.5},
                     tripcolorkw={"cmap": cmap})

    ax.margins(0)
    ax.set_aspect("equal")

    plt.xticks(np.arange(len(traits)) + .5, traits, rotation=90)
    plt.yticks(np.arange(len(annotations)) + .5, annotations)

    if not (metric == 'tau_pvalue' and turn_binary):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(r, cax=cax)
        cbar.ax.set_ylabel(metric_tex[metric])

    plt.tight_layout(rect=[0.025, 0.025, 0.975, 0.975])

    if metric == 'tau_pvalue' and turn_binary:
        metric = 'tau_pvalue_binary'

    output_fname = os.path.join(plot_dir, 'meta', 'annotations', 'heatmap',
                                f"{metric}{fig_format}")
    makedir(os.path.dirname(output_fname))

    plt.savefig(output_fname)
    plt.close()


if __name__ == '__main__':

    # --------------------- Configurations ---------------------

    ref_pop = "EUR"
    ld_estimators = ['D2']
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    sumstats_file = "data/independent_sumstats/sumstats_table.csv"
    main_regres_dir = f"results/regression/{ref_pop}"

    fig_format = '.svg'
    sns.set_context('talk')
    include_title = False

    metric_tex = {
        'hg2': '$h_g^2$',
        'LRT': 'LRT',
        'Intercept': 'Intercept',
        'tau_star': '$\\tau^*$',
        'tau_star_w': '$\\tau^*(w)$',
        'tau_star_w2': '$\\tau^*(w2)$',
        'tau': '$\\tau$',
        'enrichment': 'Enrichment',
        'enrichment2': 'Enrichment 2',
        'differential_enrichment': 'Differential Enrichment'
    }

    ld_scores_colors = {
        'D2_0.0': '#E15759',
        'D2_0.25': '#C66F6F',
        'D2_0.5': '#AC8786',
        'D2_0.75': '#919F9C',
        'D2_1.0': '#76B7B2',
        'R2_0.0': '#B07AA1',
        'R2_0.25': '#C17F84',
        'R2_0.5': '#D18466',
        'R2_0.75': '#E28949',
        'R2_1.0': '#F28E2B',
        'S-D2_0.0': '#E15759',
        'S-D2_0.25': '#C66F6F',
        'S-D2_0.5': '#AC8786',
        'S-D2_0.75': '#919F9C',
        'S-D2_1.0': '#76B7B2',
        'S-R2_0.0': '#B07AA1',
        'S-R2_0.25': '#C17F84',
        'S-R2_0.5': '#D18466',
        'S-R2_0.75': '#E28949',
        'S-R2_1.0': '#F28E2B'
    }

    method_names = {
        'D2_0.0': '$D^2 (\\alpha=0, \\theta=0)$',
        'D2_0.25': '$D^2 (\\alpha=0.25, \\theta=0)$',
        'D2_0.5': '$D^2 (\\alpha=0.5, \\theta=0)$',
        'D2_0.75': '$D^2 (\\alpha=0.75, \\theta=0)$',
        'D2_1.0': '$D^2 (\\alpha=1, \\theta=0)$',
        'R2_0.0': '$r^2 (\\alpha=0, \\theta=0)$',
        'R2_0.25': '$r^2 (\\alpha=0.25, \\theta=0)$',
        'R2_0.5': '$r^2 (\\alpha=0.5, \\theta=0)$',
        'R2_0.75': '$r^2 (\\alpha=0.75, \\theta=0)$',
        'R2_1.0': '$r^2 (\\alpha=1, \\theta=0)$',
        'S-D2_0.0': '$D^2 (\\alpha=0, \\theta=1)$',
        'S-D2_0.25': '$D^2 (\\alpha=0.25, \\theta=1)$',
        'S-D2_0.5': '$D^2 (\\alpha=0.5, \\theta=1)$',
        'S-D2_0.75': '$D^2 (\\alpha=0.75, \\theta=1)$',
        'S-D2_1.0': '$D^2 (\\alpha=1, \\theta=1)$',
        'S-R2_0.0': '$r^2 (\\alpha=0, \\theta=1)$',
        'S-R2_0.25': '$r^2 (\\alpha=0.25, \\theta=1)$',
        'S-R2_0.5': '$r^2 (\\alpha=0.5, \\theta=1)$',
        'S-R2_0.75': '$r^2 (\\alpha=0.75, \\theta=1)$',
        'S-R2_1.0': '$r^2 (\\alpha=1, \\theta=1)$'
    }

    metric_format = {
        'hg2': '{0:.3f}',
        'Intercept': '{0:.3f}',
        'LRT': '{0:.3e}',
        'tau': '{0:.3e}',
        'tau_star': '{0:.3f}',
        'tau_star_w': '{0:.3f}',
        'tau_star_w2': '{0:.3f}',
        'enrichment': '{0:.3f}',
        'enrichment2': '{0:.3f}',
        'differential_enrichment': '{0:.3e}'
    }

    alpha_str = list(map(str, alphas))
    ld_scores_ord = [m for m in method_names if 'S-' not in m and
                     m.split('_')[-1] in alpha_str and
                     m.split('_')[0] in ld_estimators]

    global_metrics = ["hg2", "Intercept", "LRT"]
    annotation_metrics = ["tau", "tau_star", "tau_star_w", "tau_star_w2",
                          "enrichment", "enrichment2",
                          "differential_enrichment"]

    trait_subset = [
        'Height (UKBB)', 'FEV1-FVC Ratio', 'Red Blood Cell Count',
        'Systolic Blood Pressure', 'BMI (UKBB)', 'Platelet Count',
        'Heel T Score', 'Forced Vital Capacity (FVC)', 'Waist-hip Ratio',
        'White Blood Cell Count', 'HDL', 'Neuroticism', 'Smoking Status',
        'College Education', 'Years of Education'
    ]

    enrichment_exclude = [
        'BLUEPRINT_DNA_methylation_MaxCPP',
        'BLUEPRINT_H3K27acQTL_MaxCPP',
        'BLUEPRINT_H3K4me1QTL_MaxCPP',
        'Backgrd_Selection_Stat',
        'CpG_Content_50kb',
        'GERP.NS',
        'GTEx_eQTL_MaxCPP',
        'Human_Enhancer_Villar_Species_Enhancer_Count',
        'MAF_Adj_ASMC',
        'MAF_Adj_LLD_AFR',
        'MAF_Adj_Predicted_Allele_Age',
        'Nucleotide_Diversity_10kb',
        'Recomb_Rate_10kb',
        'base'
    ]

    trait_meta = pd.read_csv(sumstats_file)
    trait_meta['id'] = trait_meta['File'].apply(lambda f: os.path.basename(f).replace('.sumstats.gz', ''))

    trait_name_dict = dict(trait_meta[['id', 'Trait name']].values)
    trait_source_dict = dict(trait_meta[['id', 'Source']].values)

    # ----------------------------------------------------------

    for regres_dir in glob.glob(os.path.join(main_regres_dir, '*/')):

        print(f">> Processing regression results from {regres_dir}...")

        plot_dir = regres_dir.replace('results/regression', 'figures/regression_figures_test')
        table_dir = regres_dir.replace('results/regression', 'tables/regression_tables_test')

        makedir(table_dir)

        combined_global_df = []
        combined_annot_df = []

        for trait_file in glob.glob(os.path.join(regres_dir, "*/regression_res.pbz2")):

            trait = os.path.basename(os.path.dirname(trait_file))

            print(f">> Plotting for {trait_name_dict[trait]}...")

            trait_reg_res = read_pbz2(trait_file)

            for m in ld_scores_ord:
                plot_regression_result(trait_reg_res[m],
                                       f'Regression Result for {trait_name_dict[trait]}',
                                       os.path.join(plot_dir, trait, 'regression',
                                                    f'{m}_regression' + fig_format))

            for gm in global_metrics:
                for m in ld_scores_ord:
                    combined_global_df.append({
                        'Trait': trait_name_dict[trait],
                        'Metric': gm,
                        'Method': m,
                        'Score': trait_reg_res[m][gm],
                        'Score SE': trait_reg_res[m][gm + '_se']
                    })
                    combined_global_df.append({
                        'Trait': trait_name_dict[trait],
                        'Metric': gm,
                        'Method': 'S-' + m,
                        'Score': trait_reg_res['S-' + m][gm],
                        'Score SE': trait_reg_res['S-' + m][gm + '_se']
                    })

                plot_global_metric_estimates(trait, trait_reg_res, metric=gm)

            for an in annotation_metrics:

                for m in ld_scores_ord:

                    try:
                        combined_annot_df.extend([{
                            'Trait': trait_name_dict[trait],
                            'Metric': an,
                            'Method': 'S-' + m,
                            'Annotation': annot_name,
                            'Score': annot_val,
                            'Score SE': annot_val_se
                        } for annot_name, annot_val, annot_val_se in
                            zip(trait_reg_res['S-' + m]['Annotations']['Names'],
                                trait_reg_res['S-' + m]['Annotations'][an],
                                np.abs(trait_reg_res['S-' + m]['Annotations'][an + '_se'])
                                )
                        ])
                    except KeyError:
                        combined_annot_df.extend([{
                            'Trait': trait_name_dict[trait],
                            'Metric': an,
                            'Method': 'S-' + m,
                            'Annotation': annot_name,
                            'Score': annot_val,
                            'Score SE': 0.0
                        } for annot_name, annot_val in
                            zip(trait_reg_res['S-' + m]['Annotations']['Names'],
                                trait_reg_res['S-' + m]['Annotations'][an]
                                )
                        ])

                # plot_annotation_estimates(trait, trait_reg_res, metric=an)

        combined_global_df = pd.DataFrame(combined_global_df)
        combined_global_df.to_csv(os.path.join(table_dir, "global_metrics.csv"), index=False)

        combined_annot_df = pd.DataFrame(combined_annot_df)
        combined_annot_df.to_csv(os.path.join(table_dir, "annotation_metrics.csv"), index=False)

        for mt in global_metrics:

            print(f'> Global metric {mt}')

            plot_combined_global_results(combined_global_df, mt)
            plot_combined_global_results(combined_global_df, mt, methods_included='stratified')
            plot_combined_global_results(combined_global_df, mt, methods_included='univariate')
            plot_combined_global_results(combined_global_df, mt, methods_included='figure2')

            if mt == "Intercept":
                plot_combined_global_results(combined_global_df, mt, adjust_intercept=False)
                plot_combined_global_results(combined_global_df, mt, adjust_intercept=False,
                                             methods_included='stratified')
                plot_combined_global_results(combined_global_df, mt, adjust_intercept=False,
                                             methods_included='univariate')
                plot_combined_global_results(combined_global_df, mt, adjust_intercept=False,
                                             methods_included='figure2')

            if 'R2_1.0' in ld_scores_ord:
                plot_combined_global_results(combined_global_df, mt, relative_to='R2_1.0')
                plot_combined_global_results(combined_global_df, mt, relative_to='R2_1.0',
                                             methods_included='univariate')
                plot_combined_global_results(combined_global_df, mt, relative_to='R2_1.0',
                                             methods_included='figure2')

                plot_combined_global_results(combined_global_df, mt, relative_to='S-R2_1.0')
                plot_combined_global_results(combined_global_df, mt, relative_to='S-R2_1.0',
                                             methods_included='stratified')
                plot_combined_global_results(combined_global_df, mt, relative_to='S-R2_1.0',
                                             methods_included='figure2')

            if 'D2_1.0' in ld_scores_ord:
                plot_combined_global_results(combined_global_df, mt, relative_to='S-D2_1.0')
                plot_combined_global_results(combined_global_df, mt, relative_to='S-D2_1.0',
                                             methods_included='stratified')
                plot_combined_global_results(combined_global_df, mt, relative_to='S-D2_1.0',
                                             methods_included='figure2')

        for mt in annotation_metrics:

            print(f'> Annotation metric {mt}')

            if mt not in ["enrichment", "enrichment2", "differential_enrichment"]:
                plot_meta_analyzed_annotation_coefficients_bar(combined_annot_df, mt)

                if 'R2_1.0' in ld_scores_ord:
                    plot_meta_analyzed_annotation_coefficients_bar(combined_annot_df, mt,
                                                                   relative_to='S-R2_1.0')

            if 'S-R2_1.0' not in ld_scores_ord:
                ref_ld = 'S-D2_1.0'
            else:
                ref_ld = 'S-R2_1.0'

            plot_meta_analyzed_annotation_coefficients_scatter(combined_annot_df, mt, x=ref_ld)

            for y_ldc in ['S-D2_0.25', 'S-D2_0.5', 'S-D2_0.75']:
                if y_ldc.replace('S-', '') in ld_scores_ord:
                    plot_meta_analyzed_annotation_coefficients_scatter(combined_annot_df,
                                                                       mt,
                                                                       x=ref_ld,
                                                                       y=y_ldc,
                                                                       write_table=False)

            plot_trait_vs_annotation_heatmap(combined_annot_df, mt,
                                             bottom_m=ref_ld, right_m=ref_ld)
