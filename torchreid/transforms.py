from __future__ import absolute_import
from __future__ import division

from torchvision.transforms import *
import torchvision.transforms.functional as TF
import torch

from PIL import Image
import random
import numpy as np
import math


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

class CenterCrop(object):

    def __init__(self, height, width, interpolation=Image.BILINEAR):
        assert height == width, '`CenterCrop` should output a square.'
        self.output_size = height

        self.interpolation = interpolation

    def __call__(self, image):

        from torchvision.transforms.functional import center_crop

        w, h = image.size
        image = center_crop(image, min(w, h))
        return image.resize((self.output_size, self.output_size), self.interpolation)

class CenterCropN(object):

    def __init__(self, output_size, n, m, index, interpolation=Image.BILINEAR):

        self.output_size = output_size
        self.n = n
        self.m = m
        self.index = index
        self.interpolation = interpolation

    def __call__(self, image):

        from torchvision.transforms.functional import crop

        w, h = image.size

        if w < h:
            padding = int(round((h - w) / self.m))
            top = int(round((h - w) / 2.)) + self.index * padding
            left = 0
        else:
            padding = int(round((w - h) / self.m))
            left = int(round((w - h) / 2.)) + self.index * padding
            top = 0

        image = crop(image, top, left, min(w, h), min(w, h))

        return image.resize((self.output_size, self.output_size), self.interpolation)

class RandomCenterCrop(object):

    def __init__(self, output_size, interpolation=Image.BILINEAR):

        self.output_size = output_size
        self.interpolation = interpolation

    def __call__(self, image, rnd=True):

        from torchvision.transforms.functional import crop

        w, h = image.size

        padding = abs(h - w) * 5 / 6 / 2
        offset = random.uniform(-padding, padding) if rnd else 0.
        if w < h:
            top = int(round((h - w) / 2. + offset))
            left = 0
        else:
            left = int(round((w - h) / 2. + offset))
            top = 0

        image = crop(image, top, left, min(w, h), min(w, h))

        return image.resize((self.output_size, self.output_size), self.interpolation)


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


def build_training_transforms(height, width, data_augment):

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    print('Using augmentation:', data_augment)

    transforms = []
    if 'crop' in data_augment:
        transforms.append(Random2DTranslation(height, width))
    elif 'center-crop' in data_augment:
        transforms.append(CenterCrop(height, width))
    else:
        transforms.append(Resize((height, width)))

    transforms.append(RandomHorizontalFlip())

    if 'color-jitter' in data_augment:
        transforms.append(ColorJitter())

    transforms.append(ToTensor())
    transforms.append(normalize)

    if 'random-erase' in data_augment:
        transforms.append(RandomErasing())

    return transforms


def _build_transforms(height, width, is_train, data_augment, **kwargs):
    """Build transforms

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - is_train (bool): train or test phase.
    - data_augment (str)
    """
    # use imagenet mean and std as default
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    transforms = []

    if is_train:
        transforms = build_training_transforms(height, width, data_augment)
    else:
        if 'center-crop' in data_augment:
            transforms += [CenterCrop(height, width)]
        else:
            transforms += [Resize((height, width))]

        if kwargs.get('flip', False):
            transforms += [Lambda(lambda img: TF.hflip(img))]

        transforms += [ToTensor()]
        transforms += [normalize]

    return transforms


def build_transforms(height, width, is_train, data_augment, **kwargs):

    transforms = Compose(_build_transforms(height, width, is_train, data_augment, **kwargs))
    if is_train:
        print('Using transform:', transforms)

    return transforms

def build_transforms_random_crop_n(height, width, is_train, data_augment, **kwargs):

    transforms = _build_transforms(height, width, is_train, data_augment, **kwargs)
    # n, m, index = kwargs['crop_n']
    # assert n % 2 == 1 and abs(index) <= n // 2 and height == width
    transforms = transforms[1:]
    # transforms[0] = CenterCropN(height, n, m, index)

    transforms = Compose(transforms)
    if is_train:
        print('Using transform:', transforms)

    return RandomCenterCrop(height), transforms

def build_transforms_crop_n(height, width, is_train, data_augment, **kwargs):

    transforms = _build_transforms(height, width, is_train, data_augment, **kwargs)
    n, m, index = kwargs['crop_n']
    assert n % 2 == 1 and abs(index) <= n // 2 and height == width
    transforms[0] = CenterCropN(height, n, m, index)

    transforms = Compose(transforms)
    if is_train:
        print('Using transform:', transforms)

    return transforms
