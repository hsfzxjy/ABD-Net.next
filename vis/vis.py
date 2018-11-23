#!/usr/bin/env python3

import sys
import os
import os.path as osp

current_dir = osp.abspath(osp.dirname(__file__))
resolve = lambda *parts: osp.join(current_dir, *parts)  # noqa

sys.path.insert(0, resolve('..'))
# os.chdir(resolve('..'))
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
    load_weights = resolve('..', 'data/densenet121_fc512_market_xent/densenet121_fc512_market_xent.pth.tar')


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


from cnn_layer_visualization import CNNLayerVisualization

import torch.nn as nn

# original_model = model


def flatten(f):

    if isinstance(f, nn.Sequential):
        for x in f.children():
            yield from flatten(x)
    else:
        yield f


# class FakeDN(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.features = nn.Sequential(
#             *flatten(original_model.base)
#         )
#         print(self.features, len(self.features))
#         print(original_model.base)

#     def forward(self, x):
#         return self.features(x)


# model = FakeDN()

# print(list(model.base))
# v = CNNLayerVisualization(model.features, 363, 1)
# v.visualise_layer_with_hooks()

# import cv2
from torchreid.dataset_loader import read_image


from inverted_representation import InvertedRepresentation


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='layer', type=int)
    parser.add_argument('-p', dest='path', default=None)
    parser.add_argument('-i', dest='input', default='')
    options = parser.parse_args()

    path = options.path or resolve('..', 'generated_' + str(options.layer))
    img = read_image(options.input or resolve('1.jpg'))
    img = transform_test(img)
    print(img.shape)

    for i in range(0, 1024):
        model = get_model()
        # model.features = model.base
        ir = InvertedRepresentation(model, path)
        ir.generate_inverted_image_specific_layer(img.reshape((1, *img.shape)), i, (256, 128), options.layer)
