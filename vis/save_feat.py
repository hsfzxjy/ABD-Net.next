#!/usr/bin/env python3

import sys
import os
import os.path as osp

current_dir = osp.abspath(osp.dirname(__file__))
resolve = lambda *parts: osp.join(current_dir, *parts)  # noqa

sys.path.insert(0, resolve('..'))
os.chdir(resolve('..'))
sys.path.insert(0, resolve('pytorch-cnn-visualization/src'))
print(sys.path)

from torchreid import data_manager
from torchreid import transforms as T
from torchreid import models

import torch
from torchreid.utils.iotools import check_isfile


class args:
    height = 256
    width = 128
    arch = 'densenet121'
    root = 'data'
    dataset = 'market1501'
    split_id = 0
    cuhk03_labeled = False
    cuhk03_classic_split = False
    load_weights = resolve('..', 'log/densenet121-xent-market1501/best_model.pth.tar')


use_gpu = True


transform_test = T.Compose([
    T.Resize((args.height, args.width)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = data_manager.init_imgreid_dataset(
    root=args.root, name=args.dataset, split_id=args.split_id,
    cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,
)


def get_model():

    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    if check_isfile(args.load_weights):
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))
    return model


from torchreid.dataset_loader import read_image

import cv2

img = read_image(resolve('1.jpg'))
img = transform_test(img)
img = img.reshape((1, *img.shape))

model = get_model()
model.eval()

print(model.base)

root_dir = resolve('..', 'features')
os.makedirs(root_dir, exist_ok=True)

from misc_functions import recreate_image

import h5py

f = h5py.File(resolve('..', 'features.h5'), 'w')

for index, layer in enumerate(model.base):

    img = layer(img)
    this_layer_dir = resolve(root_dir, f'layer_{index}')
    os.makedirs(this_layer_dir, exist_ok=True)

    dset = f.create_dataset(f'layer_{index}', img.shape[1:])
    dset[:, :, :] = img.data.numpy()

    for c in range(img.shape[1]):
        print(index, c)
        channel = img[0, c, :, :]
        x = channel.expand(3, *channel.shape)
        cv2.imwrite(resolve(this_layer_dir, f'{c}.jpg'), recreate_image(x.reshape((1, *x.shape))))
