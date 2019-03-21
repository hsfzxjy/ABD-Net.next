import scipy.io as io
import numpy as np


def process(fn):

    dct = io.loadmat(fn)
    distmat = dct['distmat']
    qp = dct['qp']
    gp = dct['gp']

    indices = np.argsort(distmat, axis=1)
    matches = (gp[indices] == qp[:, np.newaxis]).astype(np.int32)
    print(matches)
    print(matches.shape)

process('base_distmat.mat')
