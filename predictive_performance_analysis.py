import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import glob
import os
from utils import read_pbz2


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

methods = ['S-D2_0.0', 'S-D2_0.25', 'S-D2_0.5', 'S-D2_0.75', 'S-D2_1.0']
metric = 'Mean Difference'

annot_res = []
global_res = []
avg_chi2 = []

for trait_file in glob.glob("results/regression/EUR/M_5_50_chi2filt/*/regression_res.pbz2"):
    trait_res = read_pbz2(trait_file)
    trait_name = os.path.basename(os.path.dirname(trait_file))

    for m in methods:

        for mbin, mbin_res in trait_res[m]['Predictive Performance']['Per MAF bin'].items():
            global_res.append({
                'Trait': trait_name,
                'MAFbin': mbin,
                'Score': mbin_res[metric],
                'Method': m
            })

        avg_chi2.append({
            'Trait': trait_name,
            'Score': trait_res[m]['Predictive Performance']['Overall']['Mean Predicted Chisq'],
            'Method': m
        })

        for ann, ann_res in trait_res[m]['Annotations']['Predictive Performance'].items():
            for mbin, mbin_res in ann_res['Per MAF bin'].items():
                annot_res.append({
                    'Annotation': ann,
                    'Trait': trait_name,
                    'MAFbin': mbin,
                    'Score': mbin_res[metric],
                    'Method': m
                })

annot_res = pd.DataFrame(annot_res)
global_res = pd.DataFrame(global_res)
avg_chi2 = pd.DataFrame(avg_chi2)

print(f'Average {metric} across all traits and SNP categories:')
print(avg_chi2.groupby('Method').mean())

print('= = = = = = =')

plt.subplots(figsize=(10, 8))
sns.barplot(x='MAFbin', y='Score', hue='Method', data=global_res, ci=None,
           palette=ld_scores_colors)
plt.xlabel('MAF Decile bin')
plt.ylabel('Mean(Predicted $\chi^2$ - Observed $\chi^2$)')
plt.savefig("figures/analysis/avg_mean_diff_global.svg")
plt.close()

plt.subplots(figsize=(10, 8))
sns.barplot(x='MAFbin', y='Score', hue='Method', data=annot_res, ci=None,
            palette=ld_scores_colors)
plt.xlabel('MAF Decile bin')
plt.ylabel('Mean(Predicted $\chi^2$ - Observed $\chi^2$)')
plt.savefig("figures/analysis/avg_mean_diff_annotation.svg")
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
            data=annot_res.loc[annot_res['Annotation'].isin(highly_enriched_cats)], ci=None,
            palette=ld_scores_colors)
plt.xlabel('MAF Decile bin')
plt.ylabel('Mean(Predicted $\chi^2$ - Observed $\chi^2$)')
plt.savefig("figures/analysis/avg_mean_diff_highly_enriched_annotation.svg")
plt.close()
