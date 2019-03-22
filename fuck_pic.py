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
    # print(indices)

    errors = []

    for qidx in range(len(qp)):
        # print('index =', qidx)
        # print(qp[qidx])
        qpid = pid(qp[qidx])
        gps = gp[indices[qidx][:5]]
        gpid = [pid(x) for x in gps]
        # if len([x for x in gpid if x == qpid]) < 5:
        #     print('\n'.join(gps))
        errors.append([
            5 - len([x for x in gpid if x == qpid]),
            qidx,
            qp[qidx],
            gps
        ])

    return errors


b_error = process('base_distmat.mat')
d_error = process('dan_distmat.mat')
f_error = process('final_distmat.mat')

import os
import shutil
shutil.rmtree('pics', True)
os.makedirs('pics')

for be, fe, de in zip(b_error, f_error, d_error):

    qidx = be[1]
    qf = be[2]

    if be[0] > fe[0] + 2:
        print(qidx, be[0], de[0], fe[0])
        directory = 'pics/' + str(qidx) + '/'
        os.makedirs(directory)
        shutil.copy(qf.strip(), directory + 'query.jpg')

        for base, paths in [['baseline', be[3]], ['final', fe[3]], ['dan', de[3]]]:
            d = directory + base + '/'
            os.makedirs(d)
            for x in paths:
                shutil.copy(x.strip(), d + os.path.basename(x.strip()))
