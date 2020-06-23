"""
Author: Shadi Zabad
"""

import os
import errno
import bz2
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
