import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import glob
import os
from utils import read_pbz2, makedir


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

partitioned = False
methods = ['R2_0.0', 'R2_0.25', 'R2_0.5', 'R2_0.75', 'R2_1.0']

if partitioned:
    methods = ['S-' + m for m in methods]

metrics = [
    'Mean Difference',
    'Weighted Mean Difference',
    'Mean Squared Difference',
    'Weighted Squared Mean Difference',
    'Correlation',
    'Weighted Correlation'
    ]


annot_res = []
global_res = []
all_snps_chi2 = []

for trait_file in glob.glob("results/regression/EUR/M_5_50_chi2filt/*/*.pbz2"):
    trait_res = read_pbz2(trait_file)
    trait_name = os.path.basename(os.path.dirname(trait_file))

    for m in methods:
        if m in trait_file:

            for mbin, mbin_res in trait_res['Predictive Performance']['Per MAF bin'].items():
                for metric in metrics:
                    global_res.append({
                        'Trait': trait_name,
                        'MAFbin': mbin - 1,
                        'Metric': metric,
                        'Score': mbin_res[metric],
                        'Method': m
                    })

            for metric in metrics + ['Mean Predicted Chisq']:
                all_snps_chi2.append({
                    'Trait': trait_name,
                    'Metric': metric,
                    'Score': trait_res['Predictive Performance']['Overall'][metric],
                    'Method': m
                })

            if partitioned:
                for ann, ann_res in trait_res['Annotations']['Predictive Performance'].items():
                    for mbin, mbin_res in ann_res['Per MAF bin'].items():
                        for metric in metrics:

                            annot_res.append({
                                'Annotation': ann,
                                'Trait': trait_name,
                                'MAFbin': mbin,
                                'Metric': metric,
                                'Score': mbin_res[metric],
                                'Method': m
                            })

annot_res = pd.DataFrame(annot_res)
global_res = pd.DataFrame(global_res)
all_snps_chi2 = pd.DataFrame(all_snps_chi2)

print(f'Average {metric} across all traits and SNP categories:')
print(all_snps_chi2.groupby(['Method', 'Metric']).mean())

print('= = = = = = =')

makedir("figures/analysis/global")
makedir("figures/analysis/annotation")
makedir("figures/analysis/highly_enriched_annotation")

for metric in metrics:

    plt.subplots(figsize=(10, 8))
    sns.barplot(x='MAFbin', y='Score', hue='Method',
                data=global_res.loc[global_res['Metric'] == metric], ci=None,
                hue_order=methods,
                palette=ld_scores_colors)
    plt.xlabel('MAF Decile bin')
    plt.ylabel(metric)
    plt.savefig(f"figures/analysis/global/{metric}{fig_format}")
    plt.close()

    if partitioned:
        plt.subplots(figsize=(10, 8))
        sns.barplot(x='MAFbin', y='Score', hue='Method',
                    data=annot_res.loc[annot_res['Metric'] == metric], ci=None,
                    hue_order=methods,
                    palette=ld_scores_colors)
        plt.xlabel('MAF Decile bin')
        plt.ylabel(metric)
        plt.savefig(f"figures/analysis/annotation/{metric}{fig_format}")
        plt.close()

        highly_enriched_cats = [
            'Coding_UCSC',
            'Conserved_LindbladToh',
            'GERP.RSsup4',
            'synonymous',
            'Conserved_Vertebrate_phastCons46way',
            'Conserved_Mammal_phastCons46way',
            'Conserved_Primate_phastCons46way',
            'BivFlnk',
            'Ancient_Sequence_Age_Human_Promoter',
            'Human_Promoter_Villar_ExAC'
        ]

        plt.subplots(figsize=(10, 8))
        sns.barplot(x='MAFbin', y='Score', hue='Method',
                    hue_order=methods,
                    data=annot_res.loc[annot_res['Annotation'].isin(highly_enriched_cats) &
                                       (annot_res['Metric'] == metric)], ci=None,
                    palette=ld_scores_colors)
        plt.xlabel('MAF Decile bin')
        plt.ylabel(metric)
        plt.savefig(f"figures/analysis/highly_enriched_annotation/{metric}{fig_format}")
        plt.close()
