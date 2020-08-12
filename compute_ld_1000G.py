"""
Author: Shadi Zabad
Date: April 2020
"""

import numpy as np
import pandas as pd
import time
import sys
import os
import errno
import argparse
from pandas_plink import read_plink1_bin
from subprocess import check_call
import csv
from numba import njit, prange
from multiprocessing import Pool


def makedir(cdir):
    try:
        os.makedirs(cdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_plink_files(input_fname, chr_num):

    # Read/transform genotype matrices:
    try:
        gt_ac = read_plink1_bin(input_fname % chr_num + ".bed")
    except Exception as e:
        raise e

    gt_ac = np.abs(gt_ac.values - 2).astype(np.int64)
    ngt_ac = (gt_ac - gt_ac.mean(axis=0)) / gt_ac.std(axis=0)

    # Read the .bim file:
    try:
        gt_meta = pd.read_csv(input_fname % chr_num + ".bim",
                              names=['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], sep='\t')
    except Exception as e:
        raise e

    maf = gt_ac.sum(axis=0) / (2. * gt_ac.shape[0])
    maf = np.round(np.where(maf > .5, 1. - maf, maf), float_precision)
    gt_meta['MAF'] = maf

    gt_meta = gt_meta[['CHR', 'SNP', 'CM', 'BP', 'MAF']]

    if weights:
        sel_snp_idx = np.where(gt_meta['SNP'].isin(snp_list))[0]

        fgt_meta = gt_meta.iloc[sel_snp_idx, ].reset_index(drop=True)
        fgt_ac = gt_ac[:, sel_snp_idx]
        fngt_ac = ngt_ac[:, sel_snp_idx]

        return fgt_ac, fngt_ac, fgt_meta
    else:
        return gt_ac, ngt_ac, gt_meta


# --------------- Auxiliary Functions ---------------


@njit(parallel=True)
def numba_count(a, out, m, n):
    for i in prange(m):
        for j in prange(n):
            out[a[i, j], i] += 1


@njit(parallel=True)
def bincount2D_numba(a, bin_num=9):

    m, n = a.shape
    out = np.zeros((bin_num, m), dtype=np.int_)

    numba_count(a, out, m, n)

    return out

@njit
def d_squared_unphased(counts, n):
    """
    Implementation by Aaron Ragsdale
    """

    n1 = counts[0, :]
    n2 = counts[1, :]
    n3 = counts[2, :]
    n4 = counts[3, :]
    n5 = counts[4, :]
    n6 = counts[5, :]
    n7 = counts[6, :]
    n8 = counts[7, :]
    n9 = counts[8, :]

    numer = ((n2 * n4 - n2 ** 2 * n4 + 4 * n3 * n4 - 4 * n2 * n3 * n4 - 4 * n3 ** 2 * n4 - n2 * n4 ** 2 -
              4 * n3 * n4 ** 2 + n1 * n5 - n1 ** 2 * n5 + n3 * n5 + 2 * n1 * n3 * n5 - n3 ** 2 * n5 -
              4 * n3 * n4 * n5 - n1 * n5 ** 2 - n3 * n5 ** 2 + 4 * n1 * n6 - 4 * n1 ** 2 * n6 + n2 * n6 -
              4 * n1 * n2 * n6 - n2 ** 2 * n6 + 2 * n2 * n4 * n6 - 4 * n1 * n5 * n6 - 4 * n1 * n6 ** 2 - n2 * n6 ** 2 +
              4 * n2 * n7 - 4 * n2 ** 2 * n7 + 16 * n3 * n7 - 16 * n2 * n3 * n7 - 16 * n3 ** 2 * n7 -
              4 * n2 * n4 * n7 - 16 * n3 * n4 * n7 + n5 * n7 + 2 * n1 * n5 * n7 -
              4 * n2 * n5 * n7 - 18 * n3 * n5 * n7 - n5 ** 2 * n7 + 4 * n6 * n7 + 8 * n1 * n6 * n7 - 16 * n3 * n6 * n7 -
              4 * n5 * n6 * n7 - 4 * n6 ** 2 * n7 - 4 * n2 * n7 ** 2 - 16 * n3 * n7 ** 2 - n5 * n7 ** 2 -
              4 * n6 * n7 ** 2 + 4 * n1 * n8 - 4 * n1 ** 2 * n8 + 4 * n3 * n8 + 8 * n1 * n3 * n8 -
              4 * n3 ** 2 * n8 + n4 * n8 - 4 * n1 * n4 * n8 + 2 * n2 * n4 * n8 - n4 ** 2 * n8 -
              4 * n1 * n5 * n8 - 4 * n3 * n5 * n8 + n6 * n8 + 2 * n2 * n6 * n8 - 4 * n3 * n6 * n8 +
              2 * n4 * n6 * n8 - n6 ** 2 * n8 - 16 * n3 * n7 * n8 - 4 * n6 * n7 * n8 - 4 * n1 * n8 ** 2 -
              4 * n3 * n8 ** 2 - n4 * n8 ** 2 - n6 * n8 ** 2 + 16 * n1 * n9 - 16 * n1 ** 2 * n9 +
              4 * n2 * n9 - 16 * n1 * n2 * n9 - 4 * n2 ** 2 * n9 + 4 * n4 * n9 - 16 * n1 * n4 * n9 + 8 * n3 * n4 * n9 -
              4 * n4 ** 2 * n9 + n5 * n9 - 18 * n1 * n5 * n9 - 4 * n2 * n5 * n9 + 2 * n3 * n5 * n9 -
              4 * n4 * n5 * n9 - n5 ** 2 * n9 - 16 * n1 * n6 * n9 -
              4 * n2 * n6 * n9 + 8 * n2 * n7 * n9 + 2 * n5 * n7 * n9 - 16 * n1 * n8 * n9 - 4 * n4 * n8 * n9 -
              16 * n1 * n9 ** 2 - 4 * n2 * n9 ** 2 -
              4 * n4 * n9 ** 2 - n5 * n9 ** 2) / 16. +
             (-((n2 / 2. + n3 + n5 / 4. + n6 / 2.) * (n4 / 2. + n5 / 4. + n7 + n8 / 2.)) +
             (n1 + n2 / 2. + n4 / 2. + n5 / 4.) * (n5 / 4. + n6 / 2. + n8 / 2. + n9)) ** 2)

    return 4. * numer / (n * (n - 1) * (n - 2) * (n - 3))


# --------------------------------------------------
# --------------- LD Score Functions ---------------
# --------------------------------------------------

def compute_modified_ld_score(j, max_cm_dist=1.):

    # Obtain neighboring SNPs information:
    # --------------------------------------------
    # Condition to exclude focal snp: (gt_meta.index != gt_meta.iloc[j, ].name) &
    neighb_snps = gt_meta.loc[(np.abs(gt_meta['CM'] - gt_meta.iloc[j, ]['CM']) <= max_cm_dist), ]

    neighb_snps_annot = neighb_snps.iloc[:, annot_start_idx:].values
    neighb_snps_idx = neighb_snps.index.values
    var_xk = neighb_snps['VAR'].values
    var_xj = gt_meta.iloc[j, ]['VAR']

    # --------------------------------------------
    # Compute D^2
    gt_counts = gt_ac[:, j, np.newaxis] * 3 + gt_ac[:, neighb_snps_idx]
    count_mat = bincount2D_numba(gt_counts.T)

    # D^2 vector with all neighboring SNPs:
    D2 = d_squared_unphased(count_mat[::-1, :], N)
    D2 = (4. / var_xj) * D2

    # --------------------------------------------
    # Compute r^2
    uncr_r2 = (np.dot(ngt_ac[:, j], ngt_ac[:, neighb_snps_idx]) / N)**2
    r2 = uncr_r2 - (1. - uncr_r2)/(N - 2)

    # --------------------------------------------
    # Compute scores based on different estimators/assumptions:

    # = = = = = = D^2 based estimators = = = = = =

    scores = []

    for lds in scores_to_compute.values():

        if lds['estimator'] == 'D2':
            scores.append(
                np.dot((neighb_snps_annot * (var_xk.reshape(-1, 1)**(-lds['alpha']))).T,
                       D2)
            )
        elif lds['estimator'] == 'R2':
            scores.append(
                np.dot((neighb_snps_annot * (var_xk.reshape(-1, 1) ** (1. - lds['alpha']))).T,
                       r2)
            )
        elif lds['estimator'] == 'NR2':
            scores.append(
                np.dot((neighb_snps_annot * (var_xk.reshape(-1, 1) ** (1. - lds['alpha']))).T,
                       uncr_r2)
            )
        else:
            raise Exception(f"LD estimator {lds['estimator']} not implemented!")

    return j, scores


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LD Score Regression Using 1000 Genomes Project Data')
    parser.add_argument('--pop', dest='pop', type=str, default='EUR',
                        help='The population name')
    parser.add_argument('--weights', dest='weights', action='store_true',
                        help='Calculate the weights for the LDSC')

    args = parser.parse_args()

    # Global parameters
    # ---------------------------------------------------------
    dist_measure = "cM"
    annot_start_idx = 6
    weights = args.weights

    ld_estimator = ['D2', 'R2']  #, 'NR2']
    alpha = [0., .25, .5, .75, 1.]

    scores_to_compute = {
        lde + '_' + str(a): {
            'estimator': lde,
            'alpha': a
        }
        for lde in ld_estimator for a in alpha
    }

    population = args.pop

    # = = = = = = = = = =
    # Computational configurations:
    num_proc = 4
    float_precision = 15
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["NUMBA_NUM_THREADS"] = "2"

    # = = = = = = = = = =
    # Input:
    plink_dir = "./data/genotype_files/1000G_Phase3_" + population + "_plinkfiles/1000G." + population + ".QC.%s"
    w_snp_filter = "./data/genotype_files/w_snplist_no_MHC.snplist.bz2"
    annotations = "./data/ld_scores/1000G_Phase3_" + population + "_baselineLD_v2.2_ldscores/baselineLD.%d.annot.gz"

    # = = = = = = = = = =
    # Output:
    output_dir = "./output/ld_scores%s/1000G_Phase3_%s_mldscores/" % (['', '_weights'][weights], population)

    output_dirs = [os.path.join(output_dir, sn) for sn in scores_to_compute]
    [makedir(od) for od in output_dirs]

    # = = = = = = = = = =
    # Read the snp filter file:
    try:
        snp_list = pd.read_csv(w_snp_filter, sep="\t")['SNP'].values
    except Exception as e:
        raise e

    # ---------------------------------------------------------

    for chr_num in range(22, 0, -1):

        output_files = [os.path.join(od, "LD.%s.l2.ldscore" % str(chr_num)) for od in output_dirs]

        print("Processing chromosome %s..." % str(chr_num))

        # Read the genotype file:
        try:
            gt_ac, ngt_ac, gt_meta = read_plink_files(plink_dir, str(chr_num))
        except Exception as e:
            continue

        N, M = gt_ac.shape

        gt_meta['VAR'] = 2.*gt_meta['MAF']*(1. - gt_meta['MAF'])

        # Read the annotations file:
        if weights:
            gt_meta['base'] = 1.
        else:
            try:
                annot_df = pd.read_csv(annotations % chr_num, sep="\s+").drop(['CHR', 'BP', 'CM'], axis=1)
                gt_meta = pd.merge(gt_meta, annot_df, on='SNP')
            except Exception as exp:
                gt_meta['base'] = 1.

        output_colnames = [[cn + sn for cn in gt_meta.columns[annot_start_idx:]]
                           for sn in scores_to_compute]

        # -------------------------------------------------

        print(M, N)
        print("Computing LD Scores...")

        if not weights:
            M_tot = gt_meta.iloc[:, annot_start_idx:].sum(axis=0).values
            M_5_50 = gt_meta.loc[gt_meta['MAF'] >= .05, ].iloc[:, annot_start_idx:].sum(axis=0).values

            for of in output_files:
                np.savetxt(of.replace('.ldscore', '.M'), M_tot.reshape(1, -1), delimiter="\t", fmt='%.1f')
                np.savetxt(of.replace('.ldscore', '.M_5_50'), M_5_50.reshape(1, -1), delimiter="\t", fmt='%.1f')

        start = time.time()
        pool = Pool(num_proc)

        open_files = [open(outf, 'w') for outf in output_files]
        csv_writers = [csv.writer(outf, delimiter='\t') for outf in open_files]

        # Write the column names:
        for cw, col in zip(csv_writers, output_colnames):
            cw.writerow(list(gt_meta.columns[:annot_start_idx - 1]) + col)

        # Select the subset of snps to compute the the LD scores for:
        snps_to_process = list(np.where(gt_meta['SNP'].isin(snp_list))[0])

        # Compute the LD Scores:
        for idx, (snp_idx, ld_scores) in enumerate(pool.imap(compute_modified_ld_score, snps_to_process), 1):

            for cw, lds in zip(csv_writers, ld_scores):
                cw.writerow(list(gt_meta.iloc[snp_idx, :annot_start_idx - 1]) +
                            list(np.round(lds, float_precision)))

            if idx % 1000 == 0:
                print("Computed LD Score for %d variants" % idx)
                sys.stdout.flush()

        [outf.close() for outf in open_files]

        pool.close()
        pool.join()

        end = time.time()
        print("Processing Time:", end - start)

        # Gzip the output file
        [check_call(['gzip', '-f', of]) for of in output_files]
