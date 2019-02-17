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
    ImageDataset(dataset.gallery, transform=transform_test),
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
    parser.add_argument('-l', dest='layer', type=int)
    parser.add_argument('-p', dest='path', default=None)
    parser.add_argument('-i', dest='input', default='')
    parser.add_argument('-w', dest='weights')
    parser.add_argument('-a', dest='arch')
    parser.add_argument('--iter', type=int, default=50)
    # options = parser.parse_args()
    # args.load_weights = options.weights
    # if options.arch:
    #     args.arch = options.arch
    # print(options)

    imgs, pids, camids, _ = next(iter(testloader))
    input_img = imgs[:2]
    # input_img = input_img.view(1, *input_img.size())
    target = pids[0]

    from gradcam import GradCam
    from misc_functions import save_class_activation_on_image

    model = get_model()

    print(input_img.size())
    gradcam = GradCam(model, model.reduction_tr, 'before')
    cam = gradcam.generate_cam(input_img)

    save_class_activation_on_image(input_img, cam, 'heatmap.jpg')
