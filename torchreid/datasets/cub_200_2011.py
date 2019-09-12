import os
import glob
import re
import sys
import os.path as osp


from .bases import BaseImageDataset


class CUB_200_2011(BaseImageDataset):

    dataset_dir = 'CUB_200_2011'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)

        # self._check_before_run()

        self.relabel_mapping = None

        data = self._process_dir(self.dataset_dir, 'images.txt')
        train, val = self._get_split(data, self.dataset_dir)

        self.train = train
        self.val = val

        if verbose:
            print("=> CUB 200 2011 loaded")
            print("==> Train Set Stats")
            print('Pids', 'Images', 'Cids', sep='\t')
            print(*self.get_imagedata_info(self.train), sep='\t')
            print("==> Val Set Stats")
            print('Pids', 'Images', 'Cids', sep='\t')
            print(*self.get_imagedata_info(self.val), sep='\t')

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_train_pids = 200

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

    def _relabel(self, *datasets):

        ids = set(x[1] for dataset in datasets for x in dataset)
        mapping = {v: k for k, v in enumerate(sorted(ids))}

        new_datasets = []
        for dataset in datasets:
            new_dataset = []
            for item in dataset:
                new_item = item[:]
                new_item[1] = mapping[item[1]]
                new_dataset.append(new_item)
            new_datasets.append(new_dataset)

        return new_datasets, len(ids)

    def _process_dir(self, dir_path, id_file):

        dataset = []

        with open(osp.join(self.dataset_dir, id_file), 'r') as f:

            for line in f:
                _, fn = line.strip().split()
                id = int(fn.partition('.')[0]) - 1
                fn = osp.join(dir_path, 'images', fn)
                cid = 0
                dataset.append([fn, id, cid])

        return dataset

    def _get_split(self, data, dir_path):
        train, val = [], []
        with open(osp.join(dir_path, 'train_test_split.txt')) as f:
            for line in f:
                idx, is_train = map(int, line.strip().split())
                idx -= 1
                is_train = is_train == 1
                if is_train:
                    train.append(data[idx])
                else:
                    val.append(data[idx])

        return train, val