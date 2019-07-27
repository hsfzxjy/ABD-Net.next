import os.path as osp
from torchreid import models
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchreid.transforms import build_transforms
from torchreid.dataset_loader import read_image
import numpy as np


def load_data(files, transform):

    from glob import glob

    return [
        (transform(read_image(name)), pid, 0, name)
        for pid, name in enumerate(sorted(files))
    ]

def select_pair():
    from random import choice
    from glob import glob
    import re

    query = choice(glob('data/market1501/query'))
    pid, cid = re.findall(r'([-\d]+)_c(\d)', query)[0]
    gallery = choice(glob(f'data/market1501/bounding_box_test/{pid}_c'))

    print(query, gallery, 'selected')
    return [query, gallery]

class _args:
    arch = 'resnet50_abd_basel'
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
# parser.add_argument('-d', dest='dir')
# parser.add_argument('-o', dest='output')
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
print(set(model_dict)-set(pretrain_dict))
model.load_state_dict(model_dict)
print("Loaded pretrained weights from '{}'".format(args.weights))

model = nn.DataParallel(model).cuda()

transform = build_transforms(384, 128, is_train=False, data_augment=[])
dl = DataLoader(
    load_data(select_pair(), transform),
    batch_size=2, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
)
# gallery = DataLoader(
#     load_data(args.dir, transform),
#     batch_size=5, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
# )

def get_feature(outputs, i, position):

    return outputs[2][position][i].data.numpy(), outputs[3]['after'][position][i].data.numpy()

def get_map(fq, fg, Fg):

    import numpy as np
    from numpy.linalg import norm
    result = np.zeros(Fg.shape[1:])
    fg_norm = norm(fg)
    for i in result.shape[0]:
        for j in result.shape[1]:
            result[i][j] = fq * Fg[:,i,j]
    
    result = result / fg_norm
    max = np.max(result)

    return 1 / (1 + np.exp(-(result - max)))

def generate_map(outputs):
    fq, Fq = get_feature(outputs, 0, 0)
    fg, Fg = get_feature(outputs, 1, 0)

    import scipy.io as io
    import matplotlib.pyplot as plt
    plt.imshow(get_map(fq, fg, Fg))
    plt.savefig('a.png')

generate_map(model(next(dl)))

# with open(args.output, 'w') as f:
#     for qname, glist in sorted(list(test(model, [query], [gallery], True).items())):
#         print(qname, file=f, end='\t')
#         print(*glist, file=f, sep='\t', end='\n')
