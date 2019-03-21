import scipy.io as io
import numpy as np


def process(fn):

    dct = io.loadmat(fn)
    distmat = dct['distmat']
    qp = dct['qp']
    gp = dct['gp']

    indices = np.argsort(distmat, axis=1)
    print(indices)

    for qidx in range(10):
        print('index =', qidx)
        print(qp[qidx])
        for i in range(5):
            print(gp[indices[qidx][i]])

process('base_distmat.mat')
