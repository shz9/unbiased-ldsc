"""
Author: Shadi Zabad
"""

import os
import errno
import bz2
import numpy as np
import _pickle as cPickle


def makedir(cdir):
    try:
        os.makedirs(cdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def write_pbz2(f_name, data):
    makedir(os.path.dirname(f_name))
    with bz2.BZ2File(f_name, 'w') as f:
        cPickle.dump(data, f)


def read_pbz2(f_name):
    with bz2.BZ2File(f_name, 'rb') as f:
        data = cPickle.load(f)
    return data


def get_multiplicative_factor(mean_val):
    val = int(np.log10(np.abs(mean_val)))
    return 10.0**(val - np.sign(val))


def fig_size(hw_ratio=(np.sqrt(5) - 1.0) / 2.0, latex_colwidth_pt=345.0):

    fig_width_pt = latex_colwidth_pt
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    fig_width = fig_width_pt * inches_per_pt  # width in inches
    fig_height = fig_width * hw_ratio  # height in inches

    return fig_width, fig_height
