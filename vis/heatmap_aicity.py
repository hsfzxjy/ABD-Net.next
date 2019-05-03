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
    arch = 'resnet50_abd_old'
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
    load_weights = '0503_aicity_log/ABD_old_1_1_center_crop_re/checkpoint_ep52.pth.tar'


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
        print('Missing', set(model_dict) - set(pretrain_dict))
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


torch.random.manual_seed(0)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix')
    # parser.add_argument('arch')
    parser.add_argument('fn')
    parser.add_argument('ckpt')
    # parser.add_argument('result')
    # parser.add_argument('num', default=101, type=int)
    options = parser.parse_args()
    if options.ckpt:
        args.load_weights = options.ckpt
    # args.arch = options.arch

    # with open(options.result, 'r') as f:

    #     queries = ['data/aicity19/image_query/000001.jpg']
    #     tests = []
    #     for x in f.readline().strip().split():
    #         tests += [f'data/aicity19/image_test/{x.zfill(6)}.jpg']

    # for i, (imgs, pids, _, fns) in enumerate(train_loader):

        # if i != options.num:
        #     continue
        #
    fn = options.fn
    pids = None
    # for _, (imgs, pids, _, fns) in enumerate(train_loader):

    #     fn = fns[0]
    #     pid = pids[0].item()
    #     break

    for _ in range(1):

        orig_img = read_image(fn)
        input_img = transform_test(orig_img)
        input_img = torch.stack([input_img, deepcopy(input_img)]).cuda()
        # input_img = imgs[:2].cuda()

        from gradcam import GradCam
        from misc_functions import save_class_activation_on_image
        import cv2

        model = gradcam = cam = None

        for attrname, times, basename in [
            ('conv3', 0, 'conv3'),
            ('sum_conv1', 0, 'sum_conv0'),
            # ('sum_conv1', 1, 'sum_conv1'),
            # ('dummy_fd', 'shallow'),
            # ('fc', 'fc'),
            ('cam_module1', 0, 'cam_module0'),
            # ('cam_module1', 1, 'cam_module1'),
            ('pam_module1', 0, 'pam_module0'),
            # ('pam_module1', 1, 'pam_module1'),
            ('reduction_tr', 0, 'reduction'),

            # ('conv1', 'conv1'),
            # ('relu', 'relu'),
        ]:
            # if attrname not in options.layer.split(','):
            #     continue
            prefix = f'{options.prefix}/{fn}/{basename}/output'
            print('Making', prefix)
            del model
            del cam
            del gradcam
            input_img = transform_test(orig_img)
            input_img = input_img.view(1, *input_img.size()).cuda()
            print(input_img.size())
            # input_img = torch.stack([input_img, deepcopy(input_img)]).cuda()
            torch.cuda.empty_cache()
            model = get_model()
            gradcam = GradCam(model, getattr(model, attrname), times)
            cam = gradcam.generate_cam(input_img, pids)

            save_class_activation_on_image(cv2.imread(fn), cam, prefix)
