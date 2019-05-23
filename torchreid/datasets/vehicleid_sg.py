import os
from glob import glob
import re
import sys
import os.path as osp
from collections import defaultdict

from .bases import BaseImageDataset

class VehicleID_SG(BaseImageDataset):

    dataset_dir = 'VehicleID_V1.0'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.test_info_dir = osp.join(self.dataset_dir, 'singleG')

        datasets = {}
        for num in [800, 1600, 2400, 3200, 6000, 13164]:
            q, g = self._get_query_test(osp.join(self.test_info_dir, 'query_{}_singleG.txt'.format(num)), osp.join(self.test_info_dir, 'gallery_{}_singleG.txt'.format(num)))
            datasets[num] = dict(query=q, gallery=g)

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
            qf.readline()
            for line in qf:
                name, vid, _, _ = line.strip().split()
                name = osp.join(self.image_dir, name)
                vid = int(vid)
                qs.append((name, vid, ids[vid]))
                ids[vid] += 1

            gf.readline()
            for line in gf:
                name, vid, _, _ = line.strip().split()
                name = osp.join(self.image_dir, name)
                vid = int(vid)
                gs.append((name, vid, ids[vid]))
                ids[vid] += 1

        return qs, gs
