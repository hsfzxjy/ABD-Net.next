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
    height = 224
    width = 224
    arch = 'resnet50'
    branches = ['global', 'abd']
    abd_dan = ['cam', 'pam']
    abd_np = 2
    root = 'data'
    dataset = 'aicity19'
    shallow_cam = False
    abd_dim = 1024
    global_dim = 1024
    dropout = 0.5
    global_max_pooling = False
    abd_dan_no_head = False
    compatibility = False
    split_id = 0
    cuhk03_labeled = False
    cuhk03_classic_split = False
    load_weights = 'aicity_log/ABD_1_1/checkpoint_ep56.pth.tar'


use_gpu = True


transform_test = T.Compose([
    T.Resize((args.height, args.width)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], ),
])
dm = data_manager.ImageDataManager(use_gpu, ['aicity19'], ['aicity19'], 'data', height=224, width=224, train_sampler='RandomIdentitySampler')
train_loader, testloader_dict = dm.return_dataloaders()
# testloader = DataLoader(
#     ImageDataset(dataset.train, transform=transform_test),
#     batch_size=100, shuffle=False, num_workers=4,
#     pin_memory=True, drop_last=False
# )


def get_model():

    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent'}, use_gpu=use_gpu, args=dict(args.__dict__))
    if check_isfile(args.load_weights):
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))
    return model.cuda()


import torch.nn as nn

# original_model = model

from torchreid.dataset_loader import read_image
from copy import deepcopy


def flatten(f):

    if isinstance(f, nn.Sequential):
        for x in f.children():
            yield from flatten(x)
    else:
        yield f


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix')
    # parser.add_argument('arch')
    parser.add_argument('ckpt')
    parser.add_argument('layer')
    # parser.add_argument('result')
    parser.add_argument('num', default=101, type=int)
    options = parser.parse_args()
    if options.ckpt:
        args.load_weights = options.ckpt
    # args.arch = options.arch

    # with open(options.result, 'r') as f:

    #     queries = ['data/aicity19/image_query/000001.jpg']
    #     tests = []
    #     for x in f.readline().strip().split():
    #         tests += [f'data/aicity19/image_test/{x.zfill(6)}.jpg']

    for i, (imgs, pids, _, fns) in enumerate(train_loader):

        if i != options.num:
            continue

        # input_img = transform_test(read_image(fn))
        # input_img = torch.stack([input_img, deepcopy(input_img)]).cuda()
        input_img = imgs[:2].cuda()

        from gradcam import GradCam
        from misc_functions import save_class_activation_on_image
        import cv2

        model = gradcam = cam = None

        for attrgetter, basename in [
            (lambda m: m.branches[0][0].common_branch, 'sum_conv'),
            # ('dummy_fd', 'shallow'),
            # ('fc', 'fc'),
            (lambda m: m.branches[0][1].cam_module, 'cam_module'),
            (lambda m: m.branches[0][1].pam_module, 'pam_module'),
            (lambda m: m.branches[0][1].reduction, 'reduction'),
            # ('conv1', 'conv1'),
            # ('relu', 'relu'),
        ]:
            # if attrname not in options.layer.split(','):
            #     continue
            prefix = f'{options.prefix}/{fns[0]}/{basename}/output'
            print('Making', prefix)
            del model
            del cam
            del gradcam
            torch.cuda.empty_cache()
            model = get_model()
            print(1)
            gradcam = GradCam(model, attrgetter(model))
            print(2)
            cam = gradcam.generate_cam(input_img, pids[:2])

            save_class_activation_on_image(cv2.imread(f'data/aicity19/image_train/{fns[0]}'), cam, prefix)
