import scipy.io as io
import numpy as np


def pid(s):

    import re
    return re.findall(r'(\d{4})_')[0]


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
        qpid = pid(qp[qidx])
        gps = gp[indices[qidx][:5]]
        gpid = [pid(x) for x in gps]
        if len([x for x in gpid if x == qpid]) < 5:
            print('\n'.join(gps))


process('base_distmat.mat')
