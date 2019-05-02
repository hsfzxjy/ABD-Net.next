import os
import glob
import re
import sys
import os.path as osp

from .bases import BaseImageDataset


class AICity19Split(BaseImageDataset):

    dataset_dir = 'aicity19'

    def __init__(self, root='data', verbose=True, **kwargs):
        super().__init__()

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        self._check_before_run()

        self.relabel_mapping = None

        self.train = train = self._process_dir(self.train_dir, '1.train.txt')
        val = self._process_dir(self.train_dir, '1.val.txt')
        (train, val), num_classes = self._relabel(train, val)

        self.valsets = {
            '1': val
        }

        self.queries = {
            '2': self._process_dir(self.train_dir, '2.txt'),
            '3': self._process_dir(self.train_dir, '3.txt'),
            '4': self._process_dir(self.train_dir, '4.txt')
        }

        self.train_gallery = sum(self.queries.values(), [])

        if verbose:
            print("=> AICity19 loaded")
            print("==> Train Set Stats")
            print('Pids', 'Images', 'Cids', sep='\t')
            print(*self.get_imagedata_info(self.train), sep='\t')
            print("==> Val Set Stats")
            for name, valset in self.valsets.items():
                print('===>', name)
                print('Pids', 'Images', 'Cids', sep='\t')
                print(*self.get_imagedata_info(valset), sep='\t')
            print("==> Query Set Stats")
            for name, query in self.queries.items():
                print('===>', name)
                print('Pids', 'Images', 'Cids', sep='\t')
                print(*self.get_imagedata_info(query), sep='\t')
            print("==> Gallery Set Stats")
            print(*self.get_imagedata_info(self.train_gallery), sep='\t')

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        print(set(x[1] for x in train))
        print(set(x[1] for x in val))
        self.num_train_pids = num_classes

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

    def _process_dir(self, dir_path, id_file, relabel=False):

        dataset = []
        ids = set()

        with open(osp.join(self.dataset_dir, id_file), 'r') as f:

            for line in f:
                fn, id, cid = line.strip().split()
                fn = osp.join(dir_path, fn)
                id = int(id)
                cid = int(cid)
                dataset.append([fn, id, cid])
                ids.add(id)

        # if relabel:

        #     if False and self.relabel_mapping is not None:
        #         dct = self.relabel_mapping
        #     else:
        #         self.relabel_mapping = dct = {v: k for k, v in enumerate(sorted(ids))}
        #     for item in dataset:
        #         item[1] = dct[item[1]]

        return dataset
