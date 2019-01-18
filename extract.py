#!/usr/bin/env python3

import torch
import argparse
from torchreid import models


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='densenet121_fc512_fd_none_nohead_dan_none_nohead')
    parser.add_argument('--model', default='log/densenet121_fc512_fd_none_nohead_dan_none_nohead__crit_xent__sb___b___sl_0__fcl_False__reg_none__dropout_none__dau_crop__pp_before__size_256__0/checkpoint_ep60.pth.tar')
    parser.add_argument('--dest')
    parser.add_argument('--gpu', type=int, default=1)

    # Mock
    parser.add_argument('--root', type=str, default='data',
                        help="root path to data directory")
    parser.add_argument('-s', '--source-names', type=str, required=True, nargs='+',
                        help="source datasets (delimited by space)")
    parser.add_argument('-t', '--target-names', type=str, required=True, nargs='+',
                        help="target datasets (delimited by space)")
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (tips: 4 or 8 times number of gpus)")
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image")
    parser.add_argument('--split-id', type=int, default=0,
                        help="split index (note: 0-based)")
    parser.add_argument('--train-sampler', type=str, default='',
                        help="sampler for trainloader")
    parser.add_argument('--data-augment', type=str, choices=['none', 'crop', 'random-erase', 'color-jitter', 'crop,random-erase', 'crop,color-jitter'], default='crop')
    parser.add_argument('--train-batch-size', default=32, type=int,
                        help="training batch size")
    parser.add_argument('--test-batch-size', default=100, type=int,
                        help="test batch size")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="number of instances per identity")

    return parser.parse_args()


from args import image_dataset_kwargs
from torchreid.data_manager import ImageDataManager
import os

args = None

from collections import defaultdict


def evaluate(model, loader):

    model.eval()
    with torch.no_grad():

        pids_lst, f_512, f_1024 = [], [], []
        for _, (imgs, pids, _, _) in enumerate(loader):
            imgs = imgs.cuda()
            pids_lst.extend(pids)

            os.environ['NOFC'] = '1'
            features = model(imgs).data.cpu()
            features = features.div(torch.norm(features, p=2, dim=1, keepdim=True).expand_as(ff))
            f_1024.append(features)

            os.environ['NOFC'] = ''
            features = model(imgs).data.cpu()
            features = features.div(torch.norm(features, p=2, dim=1, keepdim=True).expand_as(ff))
            f_512.append(features)

        f_512 = torch.cat(f_512, 0)
        f_1024 = torch.cat(f_1024, 0)

        dct = defaultdict(lambda: {512: [], 1024: []})
        for pid, ff512, ff1024 in zip(pids_lst, f_512, f_1024):
            dct[pid][512].append(ff512)
            dct[pid][1024].append(ff1024)

        for mapping in dct.values():
            mapping[512] = torch.cat(mapping[512], 0).numpy()
            mapping[1024] = torch.cat(mapping[1024], 0).numpy()

            print(mapping.shape)


def main():

    global args
    args = get_args()
    use_gpu = True

    model = models.init_model(name=args.arch, num_classes=751, loss={'xent'}, use_gpu=args.gpu).cuda()

    checkpoint = torch.load(args.model)
    pretrain_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    dm = ImageDataManager(use_gpu, **image_dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    evaluate(model, testloader_dict['market1501']['query'])


if __name__ == '__main__':
    main()
