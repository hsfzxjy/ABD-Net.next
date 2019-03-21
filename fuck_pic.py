import scipy.io as io
import numpy as np


def pid(s):

    import re
    return re.findall(r'(\d{4})_', s)[0]


def process(fn):

    dct = io.loadmat(fn)
    distmat = dct['distmat']
    qp = dct['qp']
    gp = dct['gp']

    indices = np.argsort(distmat, axis=1)
    print(indices)

    errors = []

    for qidx in range(len(qp)):
        print('index =', qidx)
        print(qp[qidx])
        qpid = pid(qp[qidx])
        gps = gp[indices[qidx][:5]]
        gpid = [pid(x) for x in gps]
        # if len([x for x in gpid if x == qpid]) < 5:
        #     print('\n'.join(gps))
        errors.append([
            5 - len([x for x in gpid if x == qpid]),
            qp[qidx],
            gps
        ])

    return errors

b_error = process('base_distmat.mat')
f_error = process('final_distmat.mat')

for be, fe in zip(b_error, f_error):

    if be[0] > fe[0]:
        print(be[2], fe[2])

