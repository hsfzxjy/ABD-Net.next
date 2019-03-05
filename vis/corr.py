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
from torch.utils.data import DataLoader
from torchreid.dataset_loader import ImageDataset


class args:
    height = 384
    width = 128
    arch = 'resnet50_sf_tr4_fc1024_fd_ab_nohead_dan_cam_pam_nohead'
    root = 'data'
    dataset = 'market1501'
    split_id = 0
    cuhk03_labeled = False
    cuhk03_classic_split = False
    load_weights = '0215_trick_log/resnet50_sf_tr4_fc1024_fd_ab_nohead_dan_cam_pam_nohead__crit_singular__htri_sb_1e-6__b_1e-3__sl_-179__fcl_False__reg_const_svmo__dropout_incr__dau_crop,random-erase__pp_before,after,cam,pam,layer5__size_384__ep_200__nht__lh_.1__dr__0/checkpoint_ep118.pth.tar'


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
testloader = DataLoader(
    ImageDataset(dataset.train, transform=transform_test),
    batch_size=100, shuffle=False, num_workers=4,
    pin_memory=True, drop_last=False
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


# from inverted_representation import InvertedRepresentation, NaNError


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('prefix')
    parser.add_argument('arch')
    parser.add_argument('ckpt')
    parser.add_argument('layer')
    options = parser.parse_args()
    args.load_weights = options.ckpt
    args.arch = options.arch

    from gradcam import CamExtractor
    ids = [1, 3, 4, 5, 8, 13]
    for i, (imgs, pids, camids, img_paths) in enumerate(testloader):

        if i > 40:
            break

        if i not in ids:
            continue

        input_img = imgs[:2]

        model = None
        model = get_model()
        extractor = CamExtractor(model, getattr(model, options.layer))
        output = extractor.forward_pass(input_img)[0][0]
        output = output.view(output.size(0), -1).data.numpy()
        print(output.shape)
