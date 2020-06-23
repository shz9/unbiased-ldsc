from collections.abc import Iterable
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
import os
from utils import read_pbz2, makedir
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

plt.rcParams['hatch.linewidth'] = 2
r_meta = importr('meta')


def meta_analyze_annot(annot_df, metric='tau_star', ma_type='random'):
    meta_results = []
    meta_results_se = []

    for method in annot_df['Method'].unique():

        m_annot_df = annot_df.loc[(annot_df['Method'] == method) &
                                  (annot_df['Metric'] == metric) &
                                  (annot_df['Annotation'].isin(annot_subset))]

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
        plt.plot(x, dt['N']*dt['hg2']*x/dt['MC'] + dt['Intercept'], label=reg_label)

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

    for bar in rects2:
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

    mult_factor = 10**(int(np.log10(np.abs(
        np.mean([regression_results[m][annot_key][metric][keep_index].ravel()
                 for m in methods]))
    )))

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
                                    color=ld_scores_colors[m.replace('S-', '')],
                                    label=method_names[m],
                                    error_kw=dict(lw=1, capsize=2, capthick=1)))
            else:
                rects.append(ax.bar(x - (2 - idx)*width,
                                    bar_vals[m]['mean'][10*xidx:10*(xidx + 1)], width,
                                    color=ld_scores_colors[m.replace('S-', '')],
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
                                 adjust_intercept=True):

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

    methods = [lds for lds in ld_scores_ord
               if lds in mean_pivot.columns]
    methods += ['S-' + lds for lds in ld_scores_ord
                if 'S-' + lds in mean_pivot.columns]

    mean_dict = {
        k: [v[t] for t in trait_subset]
        for k, v in mean_pivot.to_dict().items()
    }
    se_dict = {
        k: [v[t] for t in trait_subset]
        for k, v in se_pivot.to_dict().items()
    }

    fig, ax = plt.subplots(figsize=(15, 8))

    width = 1.0
    x = np.arange(len(mean_dict[methods[0]])) * 2 * width * (1 + len(methods) // 2)

    for idx, m in enumerate(methods):
        rect = ax.bar(x - (len(methods) // 2 - idx) * width,
                      mean_dict[m],
                      width, yerr=se_dict[m],
                      label=method_names[m],
                      color=ld_scores_colors[m.replace('S-', '')],
                      error_kw=dict(lw=1, capsize=2, capthick=1))
        if m[:2] == 'S-':
            for bar in rect:
                bar.set_hatch('////')
                bar.set_edgecolor('white')

    ax.set_xticks(x)
    ax.set_xticklabels(trait_subset)

    for label in ax.get_xmajorticklabels():
        label.set_rotation(90)

    if relative_to is not None:
        plt.ylabel(f"{metric_tex[metric]} (% difference w.r.t. {relative_to})")
    elif adjust_intercept and metric == 'Intercept':
        plt.ylabel(f"Estimated Confounding (Intercept - 1.0)")
    else:
        plt.ylabel(f"Estimated {metric_tex[metric]}")

    plt.title(f"{metric_tex[metric]}")
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")

    if relative_to is None:
        if metric == "Intercept" and adjust_intercept:
            relative_to = 'value_adj'
        else:
            relative_to = 'value'
    else:
        relative_to = 'wrt_' + relative_to

    output_fname = os.path.join(plot_dir, 'meta', 'global',
                                f"{metric}_{relative_to}{fig_format}")
    makedir(os.path.dirname(output_fname))

    plt.tight_layout()
    plt.savefig(output_fname)
    plt.close()


def plot_meta_analyzed_annotation_coefficients(annotation_df, metric='tau_star', relative_to=None):

    mean_pivot, se_pivot = meta_analyze_annot(annotation_df, metric)

    if relative_to is not None:
        se_pivot = 100.0 * se_pivot.div(mean_pivot[relative_to], axis=0)
        mean_pivot = 100.0 * (mean_pivot.div(mean_pivot[relative_to], axis=0) - 1.0)

        mean_pivot.drop(relative_to, axis=1, inplace=True)
        se_pivot.drop(relative_to, axis=1, inplace=True)
        mult_factor = 1

    else:

        mult_factor = 10**(int(np.log10(np.abs(mean_pivot.values.mean()))))
        mean_pivot /= mult_factor
        se_pivot /= mult_factor

    methods = ['S-' + lds for lds in ld_scores_ord
               if 'S-' + lds in mean_pivot.columns]

    mean_dict = {
        k: [v[t] for t in annot_subset]
        for k, v in mean_pivot.to_dict().items()
    }
    se_dict = {
        k: [v[t] for t in annot_subset]
        for k, v in se_pivot.to_dict().items()
    }

    fig, ax = plt.subplots(figsize=(15, 8))

    width = 1.0
    x = np.arange(len(mean_dict[methods[0]])) * 2 * width * (1 + len(methods) // 2)

    for idx, m in enumerate(methods):
        rect = ax.bar(x - (len(methods) // 2 - idx) * width,
                      mean_dict[m],
                      width, yerr=se_dict[m],
                      label=method_names[m],
                      color=ld_scores_colors[m.replace('S-', '')],
                      error_kw=dict(lw=1, capsize=2, capthick=1))
        if m[:2] == 'S-':
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

    plt.title(f"Meta-analyzed {metric_tex[metric]}")
    plt.legend(frameon=False, bbox_to_anchor=(1.01, 0.5), loc="center left")

    if relative_to is None:
        relative_to = 'value'
    else:
        relative_to = 'wrt_' + relative_to

    plt.tight_layout()

    output_fname = os.path.join(plot_dir, 'meta', 'annotations',
                                f"{metric}_{relative_to}{fig_format}")
    makedir(os.path.dirname(output_fname))

    plt.savefig(output_fname)
    plt.close()


if __name__ == '__main__':

    # --------------------- Configurations ---------------------

    fig_format = '.svg'
    sns.set_context('paper')

    ld_scores_colors = {
        'LDSC': '#F28E2B',
        'LD2': '#E15759',
        'LD2MAF': '#76B7B2',
        'L2': '#B07AA1'
    }

    metric_tex = {
        'hg2': '$h_g^2$',
        'LRT': 'LRT',
        'Intercept': 'Intercept',
        'Ratio': 'Ratio',
        'tau_star': '$\\tau^*$',
        'tau': '$\\tau$'
    }

    method_names = {
        'LDSC': '$r^2 (\\alpha=1, \\theta=0)$',
        'S-LDSC': '$r^2 (\\alpha=1, \\theta=1)$',
        'L2': '$r^2 (\\alpha=0, \\theta=0)$',
        'S-L2': '$r^2 (\\alpha=0, \\theta=1)$',
        'L2MAF': '$r^2 (\\alpha=1, \\theta=0)$',
        'S-L2MAF': '$r^2 (\\alpha=1, \\theta=0)$',
        'LD2': '$D^2 (\\alpha=0, \\theta=0)$',
        'S-LD2': '$D^2 (\\alpha=0, \\theta=1)$',
        'LD2MAF': '$D^2 (\\alpha=1, \\theta=0)$',
        'S-LD2MAF': '$D^2 (\\alpha=1, \\theta=1)$'
    }

    ld_scores_ord = ["LDSC", "LD2MAF", "LD2", "L2"]
    global_metrics = ["hg2", "Intercept", "Ratio", "LRT"]
    annotation_metrics = ["tau", "tau_star"]
    trait_subset = [
        "Age at Menarche", "Age at Menopause", "Height",
        "BMI", "Waist-hip Ratio", "Systolic Blood Pressure",
        "High Cholesterol", "Type 2 diabetes", "Respiratory and ENT Diseases",
        "Auto Immune Traits", "Eczema", "Hypothyroidism"
    ]
    annot_subset = [
        'MAF_Adj_Predicted_Allele_Age', 'MAF_Adj_LLD_AFR',
        'Recomb_Rate_10kb', 'Nucleotide_Diversity_10kb',
        'Backgrd_Selection_Stat', 'CpG_Content_50kb', 'MAF_Adj_ASMC'
    ]

    main_regres_dir = f"results/regression/UKBB_data/"

    # ----------------------------------------------------------

    for regres_dir in glob.glob(os.path.join(main_regres_dir, '*/')):

        print(f">> Processing regression results from {regres_dir}...")

        plot_dir = regres_dir.replace('results/regression', 'figures/regression_figures')
        table_dir = regres_dir.replace('results/regression', 'tables/regression_tables')

        makedir(table_dir)

        combined_global_df = []
        combined_annot_df = []

        for trait_file in glob.glob(os.path.join(regres_dir, "*/regression_res.pbz2")):

            trait = os.path.basename(os.path.dirname(trait_file))

            print(f">> Plotting for {trait}...")

            trait_reg_res = read_pbz2(trait_file)

            for m in ld_scores_ord:
                plot_regression_result(trait_reg_res[m],
                                       f'Regression Result for {trait}',
                                       os.path.join(plot_dir, trait, 'regression',
                                                    f'{m}_regression' + fig_format))

            plot_regression_result([trait_reg_res['LDSC'], trait_reg_res['LD2MAF']],
                                   f'Regression Result for {trait}',
                                   os.path.join(plot_dir, trait, 'regression',
                                                'L2+LD2MAF_regression' + fig_format))

            for gm in global_metrics:
                for m in ld_scores_ord:
                    combined_global_df.append({
                        'Trait': trait,
                        'Metric': gm,
                        'Method': m,
                        'Score': trait_reg_res[m][gm],
                        'Score SE': trait_reg_res[m][gm + '_se']
                    })
                    combined_global_df.append({
                        'Trait': trait,
                        'Metric': gm,
                        'Method': 'S-' + m,
                        'Score': trait_reg_res['S-' + m][gm],
                        'Score SE': trait_reg_res['S-' + m][gm + '_se']
                    })

                plot_global_metric_estimates(trait, trait_reg_res, metric=gm)

            for an in annotation_metrics:

                for m in ld_scores_ord:
                    combined_annot_df.extend([{
                        'Trait': trait,
                        'Metric': an,
                        'Method': 'S-' + m,
                        'Annotation': annot_name,
                        'Score': annot_val,
                        'Score SE': annot_val_se
                    } for annot_name, annot_val, annot_val_se in
                        zip(trait_reg_res['S-' + m]['Annotations']['Names'],
                            trait_reg_res['S-' + m]['Annotations'][an],
                            trait_reg_res['S-' + m]['Annotations'][an + '_se']
                            )
                    ])

                plot_annotation_estimates(trait, trait_reg_res, metric=an)

        combined_global_df = pd.DataFrame(combined_global_df)
        combined_global_df.to_csv(os.path.join(table_dir, "global_metrics.csv"), index=False)

        combined_annot_df = pd.DataFrame(combined_annot_df)
        combined_annot_df.to_csv(os.path.join(table_dir, "annotation_metrics.csv"), index=False)

        for mt in global_metrics:
            plot_combined_global_results(combined_global_df, mt)
            if mt == "Intercept":
                plot_combined_global_results(combined_global_df, mt, adjust_intercept=False)

            plot_combined_global_results(combined_global_df, mt, relative_to='LDSC')
            plot_combined_global_results(combined_global_df, mt, relative_to='S-LDSC')
            plot_combined_global_results(combined_global_df, mt, relative_to='S-LD2MAF')

        for mt in annotation_metrics:
            plot_meta_analyzed_annotation_coefficients(combined_annot_df, mt)
            plot_meta_analyzed_annotation_coefficients(combined_annot_df, mt, relative_to='S-LDSC')
