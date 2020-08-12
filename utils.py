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
