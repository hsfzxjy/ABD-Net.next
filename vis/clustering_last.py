#!/usr/bin/env python3

import sys
import os
import os.path as osp

current_dir = osp.abspath(osp.dirname(__file__))
resolve = lambda *parts: osp.join(current_dir, *parts)  # noqa

sys.path.insert(0, resolve('..'))

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

os.chdir(resolve('..'))

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


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', dest='layer', type=int)
    parser.add_argument('-p', dest='path', default=None)
    parser.add_argument('-i', dest='input', default='')
    parser.add_argument('-k', type=float, default=1.07)
    options = parser.parse_args()

    path = options.path or resolve('..', 'generated_' + str(options.layer))
    img = read_image(options.input or resolve('1.jpg'))
    img = transform_test(img)
    img = img.view(1, *img.size())
    print(img.shape)
    model = get_model()
    x = model.features(img)
    x = (x[0].cpu().data).numpy()
    x = x.reshape(x.shape[0], -1)
    print(x.shape, type(x))

    from skfuzzy.cluster import cmeans
    u = cmeans(x.T, 2, options.k, 1e-11, 1000)[1]
    result = u.argmax(axis=0)
    print(result)
