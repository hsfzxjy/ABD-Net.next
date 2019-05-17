from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, VideoDataset
from .datasets import init_imgreid_dataset, init_vidreid_dataset
from .transforms import build_transforms
from .samplers import RandomIdentitySampler


class BaseDataManager(object):

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery'],\
            self.testdataset_dict[name]['query_flip'], self.testdataset_dict[name]['gallery_flip']


class ImageDataManager(BaseDataManager):
    """
    Image-ReID data manager
    """

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root,
                 split_id=0,
                 height=256,
                 width=128,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 data_augment='none',
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 cuhk03_labeled=False,  # use cuhk03's labeled or detected images
                 cuhk03_classic_split=False  # use cuhk03's classic split or 767/700 split
                 ):
        super(ImageDataManager, self).__init__()
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.split_id = split_id
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.num_instances = num_instances
        self.cuhk03_labeled = cuhk03_labeled
        self.cuhk03_classic_split = cuhk03_classic_split
        self.pin_memory = True if self.use_gpu else False

        # Build train and test transform functions
        transform_train = build_transforms(self.height, self.width, is_train=True, data_augment=data_augment)
        transform_test = build_transforms(self.height, self.width, is_train=False, data_augment=data_augment)
        transform_test_flip = build_transforms(self.height, self.width, is_train=False, data_augment=data_augment, flip=True)

        print("=> Initializing TRAIN (source) datasets")
        self.train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.source_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                self.train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        if self.train_sampler == 'RandomIdentitySampler':
            print('!!! Using RandomIdentitySampler !!!')
            self.trainloader = DataLoader(
                ImageDataset(self.train, transform=transform_train),
                sampler=RandomIdentitySampler(self.train, self.train_batch_size, self.num_instances),
                batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )

        else:
            self.trainloader = DataLoader(
                ImageDataset(self.train, transform=transform_train),
                batch_size=self.train_batch_size, shuffle=True, num_workers=self.workers,
                pin_memory=self.pin_memory, drop_last=True
            )

        print("=> Initializing TEST (target) datasets")
        self.testloader_dict = {name: {} for name in self.target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in self.target_names}

        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name, split_id=self.split_id, cuhk03_labeled=self.cuhk03_labeled,
                cuhk03_classic_split=self.cuhk03_classic_split
            )

            for sub_name, dct in dataset.datasets.items():
                (query, gallery) = dct['query'], dct['gallery']
                self.testloader_dict[name][sub_name] = dict(
                    query=DataLoader(
                        ImageDataset(query, transform=transform_test),
                        batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                        pin_memory=self.pin_memory, drop_last=False
                    ),

                    gallery=DataLoader(
                        ImageDataset(gallery, transform=transform_test),
                        batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                        pin_memory=self.pin_memory, drop_last=False
                    ),

                    query_flip=DataLoader(
                        ImageDataset(query, transform=transform_test_flip),
                        batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                        pin_memory=self.pin_memory, drop_last=False
                    ),

                    gallery_flip=DataLoader(
                        ImageDataset(gallery, transform=transform_test_flip),
                        batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                        pin_memory=self.pin_memory, drop_last=False
                    )
                )

            # self.testdataset_dict[name]['query'] = dataset.query
            # self.testdataset_dict[name]['gallery'] = dataset.gallery

        print("\n")
        print("  **************** Summary ****************")
        print("  train names      : {}".format(self.source_names))
        print("  # train datasets : {}".format(len(self.source_names)))
        print("  # train ids      : {}".format(self._num_train_pids))
        print("  # train images   : {}".format(len(self.train)))
        print("  # train cameras  : {}".format(self._num_train_cams))
        print("  test names       : {}".format(self.target_names))
        print("  *****************************************")
        print("\n")
