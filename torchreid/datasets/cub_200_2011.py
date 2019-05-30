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

        train = self._process_dir(self.dataset_dir, 'train.txt')
        val = self._process_dir(self.dataset_dir, 'val.txt')

        self.train = train

        self.valsets = {
            'val': val
        }

        self.queries = {
        }

        self.train_gallery = []

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
        self.num_train_pids = 200

    def _exclude_junk(self, train, val):

        train_ids = set(x[1] for x in train)
        return train, [x for x in val if x[1] in train_ids]

    def _enlarge_train_set(self, train):

        from collections import Counter
        counter = Counter(x[1] for x in train)

        new_train = []
        for fn, id, cid in train:
            img_count = counter[id]
            if img_count > 100:
                crop_num = 1
            else:
                crop_num = 180 // img_count

            for i in range(crop_num):
                new_train.append(('%s:%s' % (fn, i), id, cid))

        return new_train

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
                fn, id = line.strip().split()
                fn = osp.join(dir_path, fn)
                id = int(id)
                cid = 0
                dataset.append([fn, id - 1, cid])

        return dataset
