import os.path as osp
from torchreid import models
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torchreid.transforms import build_transforms
from torchreid.dataset_loader import read_image
import numpy as np
import cv2


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

    query = choice(glob('data/market1501/query/*'))
    pid, cid = re.findall(r'([-\d]+)_c(\d)', query)[0]
    gallery = choice(glob(f'data/market1501/bounding_box_test/{pid}_c*'))

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

parser = argparse.ArgumentParser()
parser.add_argument('-w', dest='weights')
parser.add_argument('-l', dest='lst')
parser.add_argument('-a', dest='arch')
parser.add_argument('-n', dest='name')
# parser.add_argument('-d', dest='dir')
# parser.add_argument('-o', dest='output')
args = parser.parse_args()

model = models.init_model(
    args.arch, num_classes=1,
    use_gpu=True, args=vars(_args)
)


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
    batch_size=2, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
)
# gallery = DataLoader(
#     load_data(args.dir, transform),
#     batch_size=5, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
# )

import os

def get_feature(outputs, i, position):

    return outputs[2][position][i].data.cpu().numpy(), outputs[3]['after'][position - 1][i].data.cpu().numpy()

def get_map(fq, fg, Fg):

    import numpy as np
    from numpy.linalg import norm
    from scipy.special import expit
    result = np.zeros(Fg.shape[1:])
    fg_norm = norm(fg)
    print(fq.shape, Fg.shape)
    result = fq.reshape(1, fq.shape[0]) @ Fg.reshape(Fg.shape[0], -1)
    result = result.reshape(Fg.shape[1:])    
    result = result / fg_norm
    max = np.max(result)
    print(max)
    print('done')
    return expit(result - max)

def generate_map(outputs, position):
    fq, Fq = get_feature(outputs, 0, position)
    fg, Fg = get_feature(outputs, 1, position)

    import scipy.io as io
    import matplotlib.pyplot as plt
    plt.imshow(get_map(fq, fg, Fg))
    plt.savefig('a.png')

def generate_CAM(outputs):

    from scipy.misc import imresize

    cam = np.zeros((24, 8))
    cam[:12, :] = generate_map(outputs, 1)
    print('hi')
    cam[12:, :] = generate_map(outputs, 2)
    cam = imresize(cam, (384, 128))
    
    return cam.transpose()


def save_class_activation_on_image(org_img, activation_map, prefix):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    dirname = os.path.dirname(prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # Grayscale activation map
    path_to_file = os.path.join(prefix + '_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    path_to_file = os.path.join(prefix + '_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    print(org_img.shape)
    org_img = cv2.resize(org_img, (384, 128))
    cv2.imwrite(dirname + '/Gallery.jpg', org_img)
    img_with_heatmap = .4 * np.float32(activation_heatmap) + .6 * np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join(prefix + '_Cam_On_Image.jpg')
    cv2.imwrite(path_to_file, np.uint8(255 * img_with_heatmap))
model.eval()

with open(args.lst) as f:
    for i, line in enumerate(f):
        fns = line.strip().split()
        dirname = f'hm/{i}/'
        dl = DataLoader(
            load_data(fns, transform),
            batch_size=2, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
        )
        outputs = model(next(iter(dl))[0])
        cam = generate_CAM(outputs)
        save_class_activation_on_image(
            cv2.imread(fns[1]), cam, dirname + args.name
        )
        cv2.imwrite(dirname + '/Query.jpg', cv2.imresize(cv2.imread(fns[0]), (384, 128)))


generate_map(model(next(iter(dl))[0]))

# with open(args.output, 'w') as f:
#     for qname, glist in sorted(list(test(model, [query], [gallery], True).items())):
#         print(qname, file=f, end='\t')
#         print(*glist, file=f, sep='\t', end='\n')
