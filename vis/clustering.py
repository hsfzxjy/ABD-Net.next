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
    out_dir = root_directory.rstrip('/') + '_output'
    os.makedirs(out_dir, exist_ok=True)

    pics = glob(osp.join(root_directory, '**', '**50.jpg'))
    if not pics:
        pics = glob(osp.join(root_directory, '*.jpg'))

    for i, fn in enumerate(pics):

        image = cv2.imread(fn)
        shape = image.shape
        # image = cv2.resize(image, (32, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((5, 5)) / 250
        # image = cv2.filter2D(image, -1, kernel)
        # image = 255 - image
        # image = cv2.Canny(image, 100, 200)
        cv2.imwrite(osp.join(out_dir, str(i) + '.jpg'), image)
        # if i == 8:
        #     plt.imshow(image)
        #     plt.show()

        result.append((image.flatten() / 255))

    return result


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
        dataset = np.array(load_dataset(options.ROOT))
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
    out_dir = options.ROOT.rstrip('/') + '_output'
    for x in (0, 1):
        im0 = np.average(dataset[result == x], axis=0)
        im0 = (im0.reshape(shape) * 255).astype(np.int)
        plt.imshow(im0)
        plt.show()
        cv2.imwrite(osp.join(out_dir, 'cluster_%d_avg.jpg' % x), im0, )
    with open(osp.join(out_dir, 'cluster_0.list'), 'w') as f0, open(osp.join(out_dir, 'cluster_1.list'), 'w') as f1:
        for i, x in enumerate(result):
            print(str(i) + '.jpg', file=f1 if x == 1 else f0)
    # clusterer = cluster.KMeans(2)

    # print(clusterer.fit_predict(dataset))
