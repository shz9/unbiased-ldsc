import os
import glob
from utils import read_pbz2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt


dfs = []

# Read the regression results:

for trait_file in glob.glob("../results/regression/EUR/M_chi2filt/*/S-*.pbz2"):
    trait_name = os.path.basename(os.path.dirname(trait_file))
    trait_method = os.path.basename(trait_file).replace('.pbz2', '')

    res = read_pbz2(trait_file)
    res_comp = read_pbz2(os.path.join(os.path.dirname(trait_file), 'S-D2_0.0.pbz2'))

    maf_bin_filt = np.array(['MAFbin' in n for n in res['Annotations']['Names']])

    univar_prop_hg2 = res['Annotation Sum'][0][maf_bin_filt] / (res['Annotation Sum'][0][0])
    var_prop = res_comp['Annotation Sum'][0][maf_bin_filt] / res_comp['Annotation Sum'][0][0]

    univar_enrich = univar_prop_hg2 / var_prop

    df = pd.DataFrame({
        'MAF bin': [int(x.replace('MAFbin', '')) for x in np.array(res['Annotations']['Names'])[maf_bin_filt]],
        'Score': res['Annotations']['hg2'][maf_bin_filt],
        'Enrichment': res['Annotations']['hg2'][maf_bin_filt] / (res['Counts'][0][maf_bin_filt] / res['Counts'][0][0]),
        'M Enrichment': res['Annotations']['hg2'][maf_bin_filt] / var_prop,
        'Univar Enrichment': univar_enrich
    })

    df.loc[len(df)] = [0, 1. - df['Score'].sum(),
                       (1. - df['Score'].sum()) / (
                                   (res['Counts'][0][0] - res['Counts'][0][maf_bin_filt].sum()) / res['Counts'][0][0]),
                       (1. - df['Score'].sum()) / (1. - var_prop.sum()),
                       (1. - univar_prop_hg2.sum()) / (1. - var_prop.sum())]
    df['Trait'] = trait_name
    df['Method'] = trait_method

    dfs.append(df)

maf_hg = pd.concat(dfs)
maf_hg['MAF bin'] = maf_hg['MAF bin'].astype(np.int)

# Plot the analysis results:

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
stratified = True

methods = ['D2_0.0', 'D2_0.25', 'D2_0.5', 'D2_0.75', 'D2_1.0']

if stratified:
    methods = ['S-' + m for m in methods]

data_df = maf_hg.loc[maf_hg['Method'].isin(methods)].reset_index()

plt.subplots(figsize=(12, 8))
sns.barplot(x='MAF bin', y='Enrichment', hue='Method',
            data=data_df, ci=None,
            hue_order=methods,
            palette=ld_scores_colors)
plt.xlabel('MAF bin')
plt.ylabel('Heritability Enrichment')
plt.legend(labels=[ld_scores_names[m] for m in methods], loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("figures/analysis/enrichment/heritability_enrichment.png")
plt.close()

plt.subplots(figsize=(12, 8))
sns.barplot(x='MAF bin', y='M Enrichment', hue='Method',
            data=data_df, ci=None,
            hue_order=methods,
            palette=ld_scores_colors)

plt.xlabel('MAF bin')
plt.ylabel('Functional Enrichment')
plt.legend(labels=[ld_scores_names[m] for m in methods], loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig("figures/analysis/enrichment/functional_enrichment.png")
plt.close()
