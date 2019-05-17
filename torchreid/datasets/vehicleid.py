import os
from glob import glob
import re
import sys
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset

class VehicleID(BaseImageDataset):

    dataset_dir = 'VehicleID_V1.0'

    def __init__(self, num, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.num = num
        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.test_info_dir = osp.join(self.dataset_dir, 'train_test_split')

        datasets = {}
        for seed in range(10):
            q, g = self._get_query_test(osp.join(self.test_info_dir, 'q_{}_{}.txt'.format(num, seed)), osp.join(self.test_info_dir, 'g_{}_{}.txt'.format(num, seed)))
            datasets[seed] = dict(query=q, gallery=g)

        train = []
        # if verbose:
        #     print("=> VeRi loaded")
        #     self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.datasets = datasets
        # self.query = query
        # self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        # self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _get_query_test(self, qfn, gfn):

        ids = defaultdict(int)
        qs = []
        gs = []
        with open(qfn) as qf, open(gfn) as gf:
            for line in qf:
                name, vid = line.strip().split()
                name = osp.join(self.image_dir, name + '.jpg')
                vid = int(vid)
                qs.append(name, vid, ids[vid])
                ids[vid] += 1

            for line in gf:
                name, vid = line.strip().split()
                name = osp.join(self.image_dir, name + '.jpg')
                vid = int(vid)
                gs.append(name, vid, ids[vid])
                ids[vid] += 1

        return qs, gs
