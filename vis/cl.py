#!/usr/bin/env python3

import cv2
import argparse
from glob import glob
import os.path as osp
import os
import numpy as np
from sklearn import cluster

import matplotlib.pyplot as plt

shape = None


def load_dataset(root_directory):

    global shape
    result = []
    nowrite = root_directory.rstrip('/').endswith('_output')
    if nowrite:
        out_dir = root_directory
    else:
        out_dir = root_directory.rstrip('/') + '_output'
        os.makedirs(out_dir, exist_ok=True)

    pics = glob(osp.join(root_directory, '**', '**50.jpg'))
    if not pics:
        pics = glob(osp.join(root_directory, '*.jpg'))

    dct = {}
    s = set(range(256))

    for i, fn in enumerate(pics):

        image = cv2.imread(fn)
        import re
        if nowrite:
            j = int(re.findall(r'(\d+)\.', fn)[-1])
        else:
            j = int(re.findall(r'/(\d+)/', fn)[0])
        s = s - set([j])
        # print(fn)
        shape = image.shape
        # image = cv2.resize(image, (32, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((5, 5)) / 250
        # image = cv2.filter2D(image, -1, kernel)
        # image = 255 - image
        # image = cv2.Canny(image, 100, 200)
        if not nowrite:
            cv2.imwrite(osp.join(out_dir, str(j) + '.jpg'), image)
        # if i == 8:
        #     plt.imshow(image)
        #     plt.show()
        if image.sum() > 1e7:
            result.append(image.flatten() / 255)
            dct[len(result) - 1] = int(j)
        else:
            print(j, 'omitted')

    print(s)
    # result.sort()
    # result = [x[1] for x in result]
    return result, dct, out_dir


def load_from_h5(fn, dset):

    import h5py

    f = h5py.File(fn, 'r')
    arr = f[dset][:, :, :]
    C, H, W = arr.shape
    return arr.reshape((C, H * W))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ROOT')
    parser.add_argument('-d', dest='dset', default='')
    parser.add_argument('-k', dest='k', default=1.07, type=float)
    options = parser.parse_args()

    if options.ROOT.endswith('.h5'):
        dataset = load_from_h5(options.ROOT, options.dset)
        options.ROOT += 'output/' + options.dset
    else:
        dataset, dct, out_dir = load_dataset(options.ROOT)
        dataset = np.array(dataset)
    # from skimage.measure import compare_ssim as ssim
    # from itertools import product

    # sims = []

    # for x, y in product(dataset, dataset):
    #     sims.append(ssim(x, y))

    # plt.imshow(np.array(sims).reshape((len(dataset),) * 2))
    # plt.show()
    from skfuzzy.cluster import cmeans
    print(dataset.shape)
    u = cmeans(dataset.T, 2, options.k, 1e-11, 700)[1]
    result = u.argmax(axis=0)
    print(result)
    print(result.sum(), (1 - result).sum())
    if result.sum() <= len(result) / 2:
        target = 1
    else:
        target = 0
    indices = [i for i, x in enumerate(result) if x == target]
    target_indices = sorted([dct[i] for i in indices])
    nontarget_indices = sorted(set(range(256)) - set(target_indices))
    # out_dir = options.ROOT.rstrip('/') + '_output'
    # for x in (0, 1):
    #     im0 = np.average(dataset[result == x], axis=0)
    #     im0 = (im0.reshape(shape) * 255).astype(np.int)
    #     plt.imshow(im0)
    #     plt.show()
    #     cv2.imwrite(osp.join(out_dir, 'cluster_%d_avg.jpg' % x), im0, )
    print(target_indices)
    print(nontarget_indices)
    with open(osp.join(out_dir, 'cluster_0.list'), 'w') as f0:
        for i in target_indices:
            print(str(i) + '.jpg', file=f0)
    #
    with open(osp.join(out_dir, 'cluster_1.list'), 'w') as f0:
        for i in nontarget_indices:
            print(str(i) + '.jpg', file=f0)

    # clusterer = cluster.KMeans(2)

    # print(clusterer.fit_predict(dataset))
