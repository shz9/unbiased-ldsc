import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
import errno


def makedir(output_dir):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def compare_ldsc(chr_num, joint_df, annot, output_fname,
                 base_score="L2_bs",
                 compare_scores=[('LD2', '$D^2$'), ('LD2MAF', '$D^2 MAF$'), ('L2_rg', '$r^2$')]):

    plt.figure(figsize=(10, 8))

    for sc, lb in compare_scores:
        sns.scatterplot(x=annot + base_score, y=annot + sc, data=joint_df, label=lb)

    # Plots line
    x = np.linspace(0.0, joint_df[[annot + sc for sc, _ in compare_scores] + [annot + base_score]].max().max(), 1000)
    plt.plot(x, x, linestyle='--', color='r')

    plt.xlabel("$r^2$ LD Score (BS)")
    plt.ylabel("$r^2, D^2$ LD Scores (Ours)")
    plt.title(annot + " (Chr " + chr_num + ")")

    plt.savefig(output_fname)
    plt.close()


def compare_ldsc_our(chr_num, mldsc_df, annot, output_fname,
                     compare_scores=[('LD2', '$D^2$'), ('LD2MAF', '$D^2 MAF$')]):

    for sc, lb in compare_scores:

        slope, intercept, r_value, p_value, std_err = stats.linregress(mldsc_df[annot + 'L2'],
                                                                       mldsc_df[annot + sc])

        fig, ax = plt.subplots(figsize=(10, 8))
        sct = ax.scatter(mldsc_df[annot + "L2"], mldsc_df[annot + sc],
                         c=mldsc_df['MAF'], cmap="cool", alpha=0.75)
        x = np.linspace(0.0, mldsc_df[[annot + "L2", annot + sc]].max().max(), 1000)
        plt.plot(x, x, linestyle='--', color='r')

        plt.text(0.05, 0.95,
                 "Slope:" + str(round(slope, 2)) + ", $R^2$:" + str(round(r_value**2, 2)),
                 verticalalignment='top', transform=ax.transAxes)

        plt.xlabel("$r^2$ LD Score (Ours)")
        plt.ylabel(lb + " LD Score (Ours)")
        plt.title(annot + " (Chr " + chr_num + ")")

        cbar = plt.colorbar(sct)
        cbar.ax.set_ylabel('MAF')

        plt.savefig(output_fname.replace(".png", sc + ".png"))
        plt.close()


weights = True

output_dir = "./plots/comparing_scores_MAF%s/" % ['', '_weights'][weights]
annot_start_index = 3

if weights:
    ldsc_file= "./reference/1000G_Phase3_weights_hm3_no_MHC/weights.hm3_noMHC.%d.l2.ldscore.gz"
    mldsc_file = "./output/Phase3_mldscores_MAF_weights/EUR/w_D2_%d.l2.ldscore.gz"
else:
    ldsc_file = "./reference/1000G_Phase3_baselineLD_v2.2_ldscores/baselineLD.%d.l2.ldscore.gz"
    mldsc_file = "./output/Phase3_mldscores_MAF/EUR/D2_%d.l2.ldscore.gz"

for chr_num in range(5, 23):

    print("Chromosome:", chr_num)

    try:
        mldsc_df = pd.read_csv(mldsc_file % chr_num, sep="\t")
        ldsc_df = pd.read_csv(ldsc_file % chr_num, sep="\t")
        ldsc_df = ldsc_df.rename(columns={'L2': 'baseL2'})
    except Exception as e:
        print(str(e))
        continue

    joint_df = pd.merge(ldsc_df, mldsc_df, on='SNP', suffixes=('_bs', '_rg'))

    print("LDSC:", len(ldsc_df))
    print("MLDSC:", len(mldsc_df))
    print("Joint:", len(joint_df))
    print("LD Score Difference:")
    print("Mean:", np.mean(joint_df['baseL2_bs'] - joint_df['baseL2_rg']),
          "Median:", np.median(joint_df['baseL2_bs'] - joint_df['baseL2_rg']),
          "Min:", np.min(joint_df['baseL2_bs'] - joint_df['baseL2_rg']),
          "Max:", np.max(joint_df['baseL2_bs'] - joint_df['baseL2_rg']))

    annots = ['base'] #[c.replace("L2", "") for c in ldsc_df.columns[annot_start_index:]]

    for a in annots:
        print(a)
        makedir(os.path.join(output_dir, "ours", a))
        makedir(os.path.join(output_dir, "bs_vs_rg", a))

        compare_ldsc_our(str(chr_num), mldsc_df, a, os.path.join(output_dir, "ours", a, str(chr_num) + ".png"))
        compare_ldsc(str(chr_num), joint_df, a, os.path.join(output_dir, "bs_vs_rg", a, str(chr_num) + ".png"))

    print("--------------")

