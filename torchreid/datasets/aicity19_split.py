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

        self.train = train = self._process_dir(self.train_dir, 'train_with_cam_1.txt', relabel=True)
        self.new_vid_old_cid_val = new_vid_old_cid_val = self._process_dir(self.train_dir, 'val_new_person_old_cam.txt')
        self.new_vid_new_cid_val = new_vid_new_cid_val = self._process_dir(self.train_dir, 'val_new_person_new_cam.txt')
        self.new_vid_old_cid_query = new_vid_old_cid_query = self._process_dir(self.train_dir, 'test_new_person_old_cam.txt')
        self.new_vid_new_cid_query = new_vid_new_cid_query = self._process_dir(self.train_dir, 'test_new_person_new_cam.txt')
        self.train_gallery = train_gallery = self._process_dir(self.train_dir, 'train_id_with_cam.txt')

        if verbose:
            print("=> AICity19 loaded")
            print("==> New VID Old CID")
            self.print_dataset_statistics(train, new_vid_old_cid_query, train_gallery)
            print("==> New VID New CID")
            self.print_dataset_statistics(train, new_vid_new_cid_query, train_gallery)
            print("===> New VID Old CID Val")
            print(self.get_imagedata_info(new_vid_old_cid_val))
            print("===> New VID New CID Val")
            print(self.get_imagedata_info(new_vid_new_cid_val))

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

        if relabel:
            dct = {v: k for k, v in enumerate(sorted(ids))}
            for item in dataset:
                item[1] = dct[item[1]]

        return dataset


