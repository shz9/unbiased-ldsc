import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import glob
import os
from utils import read_pbz2, makedir
import sys


fig_format = ".png"

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

ld_scores_names = {
        'D2_0.0': '$\\alpha=0$',
        'D2_0.25': '$\\alpha=0.25$',
        'D2_0.5': '$\\alpha=0.5$',
        'D2_0.75': '$\\alpha=0.75$',
        'D2_1.0': '$\\alpha=1$',
        'R2_0.0': '$\\alpha=0$',
        'R2_0.25': '$\\alpha=0.25$',
        'R2_0.5': '$\\alpha=0.5$',
        'R2_0.75': '$\\alpha=0.75$',
        'R2_1.0': '$\\alpha=1$',
        'S-D2_0.0': '$\\alpha=0$',
        'S-D2_0.25': '$\\alpha=0.25$',
        'S-D2_0.5': '$\\alpha=0.5$',
        'S-D2_0.75': '$\\alpha=0.75$',
        'S-D2_1.0': '$\\alpha=1$',
        'S-R2_0.0': '$\\alpha=0$',
        'S-R2_0.25': '$\\alpha=0.25$',
        'S-R2_0.5': '$\\alpha=0.5$',
        'S-R2_0.75': '$\\alpha=0.75$',
        'S-R2_1.0': '$\\alpha=1$',
    }

sns.set_context('talk')
plot_global_lines = False
partitioned = True
use_normalized_weights = True
methods = ['D2_0.0', 'D2_0.25', 'D2_0.5', 'D2_0.75', 'D2_1.0']
reference_model = 'D2_1.0'

if partitioned:
    methods = ['S-' + m for m in methods]
    reference_model = 'S-' + reference_model

# 'Predictive Performance'
performance_category = 'Predictive Performance' #f'Predictive Performance ({reference_model} Weights)'

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

metrics = [
    'Mean Difference',
    'Mean Squared Difference',
    'Correlation',
    'Weighted Mean Difference',
    'Weighted Correlation',
    'Weighted Mean Squared Difference',
    'Weighted Mean Difference (RWLS Weights)',
    'Weighted Correlation (RWLS Weights)',
    'Weighted Mean Squared Difference (RWLS Weights)',
    f'Weighted Mean Difference ({reference_model} Weights)',
    f'Weighted Correlation ({reference_model} Weights)',
    f'Weighted Mean Squared Difference ({reference_model} Weights)',
    f'Weighted Mean Difference ({reference_model} RWLS Weights)',
    f'Weighted Correlation ({reference_model} RWLS Weights)',
    f'Weighted Mean Squared Difference ({reference_model} RWLS Weights)'
    ]

exclude_traits = []

annot_res = []
global_res = []
all_snps_chi2 = []


for trait_file in glob.glob(f"results/regression/EUR/M_5_50_chi2filt/*/*.pbz2"):
    trait_res = read_pbz2(trait_file)
    trait_name = os.path.basename(os.path.dirname(trait_file))
    trait_method = os.path.basename(trait_file).replace('.pbz2', '')

    if trait_name in exclude_traits:
        continue

    for m in methods:
        if m == trait_method:

            if m == reference_model:
                perf_cat = performance_category.split('(')[0].strip()
            else:
                perf_cat = performance_category

            for mbin, mbin_res in trait_res[perf_cat]['Per MAF bin'].items():
                for metric in metrics:

                    if m == reference_model and m in metric:
                        if 'RWLS' in metric:
                            score = metric.replace(reference_model + ' ', '')
                        else:
                            score = metric.split('(')[0].strip()
                    else:
                        score = metric

                    if use_normalized_weights and 'Weighted' in score:
                        score = '(Normalized) ' + score

                    global_res.append({
                        'Trait': trait_name,
                        'MAFbin': mbin,
                        'Metric': metric,
                        'Method': m,
                        'Score': mbin_res[score]
                    })

            for metric in metrics + ['Mean Predicted Chisq', 'LRT']:
                all_snps_chi2.append({
                    'Trait': trait_name,
                    'Metric': metric,
                    'Method': m,
                })

                if metric == 'LRT':
                    all_snps_chi2[-1]['Score'] = trait_res['LRT']
                else:

                    if m == reference_model and m in metric:
                        if 'RWLS' in metric:
                            score = metric.replace(reference_model + ' ', '')
                        else:
                            score = metric.split('(')[0].strip()
                    else:
                        score = metric

                    all_snps_chi2[-1]['Score'] = trait_res[perf_cat]['Overall'][score]

            if partitioned and performance_category == 'Predictive Performance':
                for ann, ann_res in trait_res['Annotations'][perf_cat].items():
                    for mbin, mbin_res in ann_res['Per MAF bin'].items():
                        for metric in metrics:

                            if m == reference_model and m in metric:
                                if 'RWLS' in metric:
                                    score = metric.replace(reference_model + ' ', '')
                                else:
                                    score = metric.split('(')[0].strip()
                            else:
                                score = metric

                            if use_normalized_weights and 'Weighted' in score:
                                score = '(Normalized) ' + score

                            annot_res.append({
                                'Annotation': ann,
                                'Trait': trait_name,
                                'MAFbin': mbin,
                                'Metric': metric,
                                'Method': m,
                                'Score': mbin_res[score]
                            })

annot_res = pd.DataFrame(annot_res)
global_res = pd.DataFrame(global_res)
all_snps_chi2 = pd.DataFrame(all_snps_chi2)

mean_across_traits = all_snps_chi2.groupby(['Method', 'Metric']).mean()
print(f'Average metrics across all traits and SNP categories:')
print(mean_across_traits)

makedir(f"analysis/{['', 'partitioned/'][partitioned]}{performance_category}/binned")

for metric in metrics:
    all_snps_chi2.loc[all_snps_chi2['Metric'] == metric].pivot(
        index='Method', columns='Trait', values='Score'
    ).to_excel(f"analysis/{['', 'partitioned/'][partitioned]}{performance_category}/{metric}.xls")

    global_res.loc[global_res['Metric'] == metric].pivot_table(
        index=['Method', 'MAFbin'],
        columns='Trait',
        values='Score'
    ).to_excel(f"analysis/{['', 'partitioned/'][partitioned]}{performance_category}/binned/{metric}.xls")

print('= = = = = = =')

makedir(f"figures/analysis/{['', 'partitioned/'][partitioned]}{performance_category}/global")
makedir(f"figures/analysis/{['', 'partitioned/'][partitioned]}{performance_category}/annotation")
makedir(f"figures/analysis/{['', 'partitioned/'][partitioned]}{performance_category}/highly_enriched_annotation")

for metric in metrics:

    plt.subplots(figsize=(10, 8))
    sns.barplot(x='MAFbin', y='Score', hue='Method',
                data=global_res.loc[global_res['Metric'] == metric], ci=None,
                hue_order=methods,
                palette=ld_scores_colors)
    plt.xlabel('MAF bin')
    plt.ylabel(metric.split("(")[0].strip().replace('Difference', 'Error') + " ($\\chi^2$)")
    if plot_global_lines:
        for m in np.unique(global_res['Method']):
            global_val = mean_across_traits.loc[[(m, metric)], 'Score'].values[0]
            plt.axhline(global_val,
                        ls='--', color=ld_scores_colors[m],
                        label=f"{m}: {global_val:.3e}")
    plt.legend(labels=[ld_scores_names[m] for m in methods])
    plt.ylim(-.18, .23)
    plt.tight_layout()
    plt.savefig(f"figures/analysis/{['', 'partitioned/'][partitioned]}{performance_category}/global/{metric}{fig_format}")
    plt.close()

    if partitioned and performance_category == 'Predictive Performance':
        plt.subplots(figsize=(10, 8))
        sns.barplot(x='MAFbin', y='Score', hue='Method',
                    data=annot_res.loc[annot_res['Metric'] == metric], ci=None,
                    hue_order=methods,
                    palette=ld_scores_colors)
        plt.xlabel('MAF bin')
        plt.ylabel(metric.split("(")[0].strip().replace('Difference', 'Error') + " ($\\chi^2$)")
        plt.legend(labels=[ld_scores_names[m] for m in methods])
        plt.ylim(-.18, .23)
        plt.tight_layout()
        plt.savefig(f"figures/analysis/{['', 'partitioned/'][partitioned]}{performance_category}/annotation/{metric}{fig_format}")
        plt.close()

        plt.subplots(figsize=(10, 8))
        sns.barplot(x='MAFbin', y='Score', hue='Method',
                    hue_order=methods,
                    data=annot_res.loc[annot_res['Annotation'].isin(highly_enriched_cats) &
                                       (annot_res['Metric'] == metric)], ci=None,
                    palette=ld_scores_colors)
        plt.xlabel('MAF bin')
        plt.ylabel(metric.split("(")[0].strip().replace('Difference', 'Error') + " ($\\chi^2$)")
        plt.legend(labels=[ld_scores_names[m] for m in methods])
        plt.ylim(-.18, .23)
        plt.tight_layout()
        plt.savefig(f"figures/analysis/{['', 'partitioned/'][partitioned]}{performance_category}/highly_enriched_annotation/{metric}{fig_format}")
        plt.close()
