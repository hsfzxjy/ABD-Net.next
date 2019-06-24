import os
from glob import glob
import re
import sys
import os.path as osp
import numpy as np
from collections import defaultdict

from .bases import BaseImageDataset

def poly_area(pts):
    pts = np.array(pts)
    return -0.5 * (np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))


polys = [
    [6, 1, 2, 17, 15, 14, 11, 8],
    [7, 5, 6, 8],
    [7, 8, 14, 13],
    [16, 15, 17, 18],
    [18, 19, 17, 20],
    [5, 7, 12, 13, 16, 18, 4, 3],
    [14, 11, 17, 15],
    [12, 13, 16, 18],
]

def calc_feat(vec):

    from collections import defaultdict
    dct = defaultdict()
    for i in range(20):
        dct[i + 1] = [vec[2 * i], vec[2 * i + 1]]
    res = np.zeros(len(polys), dtype=np.float32)
    for i, poly in enumerate(polys):
        points = np.array([dct[k] for k in poly])
        if (points == -1).any():
            continue
        area = abs(poly_area(points))
        res[i] = area

    n = np.linalg.norm(res)
    if n != 0:
        res = res / n

    return res

class VeRiSur(BaseImageDataset):

    dataset_dir = 'veri'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()
        self.load_keypoints()
        self.sur_dim = len(polys)

        train = self._get_train()
        query, gallery = self._get_query_test()
        if verbose:
            print("=> VeRi loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

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

    def load_keypoints(self):

        def load_into_dict(fn):
            dct = {}

            with open(fn) as f:
                for line in f:
                    path, *nums = line.strip().split()
                    path = os.path.basename(path)
                    dct[path] = [int(x) for x in nums]

            return dct

        self.train_keypoints = load_into_dict(osp.join(self.dataset_dir, 'keypoint_train.txt'))
        self.test_keypoints = load_into_dict(osp.join(self.dataset_dir, 'keypoint_test.txt'))

    def _get_train(self):

        files = glob(osp.join(self.train_dir, '*'))

        ids = set()
        fns = defaultdict(list)
        for file in files:
            id, cid = map(int, re.findall(r'/(\d{4})_c(\d{3})', file)[0])
            ids.add(id)
            fns[id].append((file, cid))

        mapping = {v: i for i, v in enumerate(sorted(ids))}

        dataset = []
        missing = 0
        for k, fs in fns.items():
            for (f, cid) in fs:
                try:
                    vec = calc_feat(self.train_keypoints[osp.basename(f)])
                except KeyError:
                    missing += 1
                    continue
                dataset.append((f, mapping[k], cid, vec))
        print('Train missing:', missing)

        return dataset

    def _get_query_test(self):

        q = []
        with open(osp.join(self.dataset_dir, 'info/query_info.txt')) as f:
            f.readline()
            for line in f:
                img, pid, cid, _ = line.strip().split()
                vec = calc_feat(self.test_keypoints[osp.basename(img)])
                q.append((osp.join(self.query_dir, img), int(pid), int(cid), vec))

        g = []
        with open(osp.join(self.dataset_dir, 'info/gallery_info.txt')) as f:
            f.readline()
            for line in f:
                img, pid, cid, _ = line.strip().split()
                vec = calc_feat(self.test_keypoints[osp.basename(img)])
                g.append((osp.join(self.gallery_dir, img), int(pid), int(cid), vec))

        return q, g

        # q_files = set(osp.basename(x) for x in glob(osp.join(self.query_dir, '*')))
        # t_files = glob(osp.join(self.gallery_dir, '*'))

        # q_dataset = []
        # t_dataset = []

        # id_mapping = defaultdict(int)

        # for f in t_files:
        #     id, cid = map(int, re.findall(r'/(\d{4})_c(\d{3})', f)[0])
        #     bn = osp.basename(f)

        #     if bn in q_files:
        #         q_dataset.append((f, id, cid))
        #     t_dataset.append((f, id, cid))

        #     id_mapping[id] += 1

        # return q_dataset, t_dataset
