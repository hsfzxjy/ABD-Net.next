import os.path as osp
from torchreid import models
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchreid.transforms import build_transforms
from torchreid.dataset_loader import read_image
import numpy as np


def load_data(directory, transform):

    from glob import glob

    return [
        (transform(read_image(name)), pid, 0, name)
        for pid, name in enumerate(sorted(glob(directory + '/*.jpg')))
    ]


class _args:
    arch = 'resnet50'
    branches = ['global', 'abd']
    abd_np = 2
    abd_dan = ['cam', 'pam']
    global_dim = 1024
    abd_dim = 1024
    dropout = 0.5
    compatibility = False
    abd_dan_no_head = False
    abd_no_reduction = False
    global_max_pooling = False
    a3m_type = None
    shallow_cam = True
    resnet_last_stride = 1


model = models.init_model(
    'resnet50', num_classes=1,
    use_gpu=True, args=vars(_args)
)

parser = argparse.ArgumentParser()
parser.add_argument('-w', dest='weights')
parser.add_argument('-d', dest='dir')
args = parser.parse_args()

try:
    checkpoint = torch.load(args.weights)
except Exception as e:
    print(e)
    checkpoint = torch.load(args.weights, map_location={'cuda:0': 'cpu'})

pretrain_dict = checkpoint['state_dict']
model_dict = model.state_dict()
pretrain_dict = {
    k: v for k, v in pretrain_dict.items()
    if k in model_dict and model_dict[k].size() == v.size()
}
model_dict.update(pretrain_dict)
model.load_state_dict(model_dict)
print("Loaded pretrained weights from '{}'".format(args.weights))

model = nn.DataParallel(model).cuda()

transform = build_transforms(384, 128, is_train=False, data_augment=[])
query = DataLoader(
    load_data(args.dir, transform),
    batch_size=5, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
)
gallery = DataLoader(
    load_data(args.dir, transform),
    batch_size=5, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
)


def test(model, queryloader, galleryloader, use_gpu):

    flip_eval = False

    if flip_eval:
        print('# Using Flip Eval')

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, q_paths = [], [], [], []

        if flip_eval:
            enumerator = enumerate(zip(queryloader[0], queryloader[1]))
        else:
            enumerator = enumerate(queryloader[0])

        for batch_idx, package in enumerator:

            if flip_eval:
                (imgs0, pids, camids, paths), (imgs1, _, _, _) = package
                if use_gpu:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                features = (model(imgs0)[0] + model(imgs1)[0]) / 2.0
                # print(features.size())
            else:
                (imgs, pids, camids, paths) = package
                if use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs)[0]

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            q_paths.extend(paths)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, g_paths = [], [], [], []
        if flip_eval:
            enumerator = enumerate(zip(galleryloader[0], galleryloader[1]))
        else:
            enumerator = enumerate(galleryloader[0])

        for batch_idx, package in enumerator:
            # print('fuck')

            if flip_eval:
                (imgs0, pids, camids, paths), (imgs1, _, _, _) = package
                if use_gpu:
                    imgs0, imgs1 = imgs0.cuda(), imgs1.cuda()
                features = (model(imgs0)[0] + model(imgs1)[0]) / 2.0
                # print(features.size())
            else:
                (imgs, pids, camids, _) = package
                if use_gpu:
                    imgs = imgs.cuda()

                features = model(imgs)[0]

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            g_paths.extend(paths)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    from collections import defaultdict
    result = defaultdict(list)
    indices = np.argsort(distmat, axis=1)
    for qidx in range(distmat.shape[0]):
        order = indices[qidx]
        cnt = 0
        for gid in order:
            if gid == qidx:
                continue
            result[osp.basename(q_paths[qidx])].append(
                osp.basename(g_paths[gid]))
            cnt += 1
            if cnt == 10:
                break

    return result


for qname, glist in sorted(list(test(model, [query], [gallery], True).items())):
    print(qname, end='\t')
    print(*glist, sep='\t', end='\n')
