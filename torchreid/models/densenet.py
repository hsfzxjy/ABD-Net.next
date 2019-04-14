from __future__ import absolute_import
from __future__ import division

__all__ = ['densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512']

from collections import OrderedDict
import math
import re
import os

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F
import torchvision

from copy import deepcopy

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class DummyFD(nn.Module):

    def __init__(self, fd_getter):

        super().__init__()
        self.fd_getter = fd_getter

    def forward(self, x):

        B, C, H, W = x.shape

        for cs, cam in self.fd_getter().cam_modules:
            # try:
            #     c_tensor = torch.tensor(cs).cuda()
            # except RuntimeError:
            c_tensor = torch.tensor(cs).cuda()

            new_x = x[:, c_tensor]
            new_x = cam(new_x)
            x[:, c_tensor] = new_x

        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(
            num_input_features, bn_size *
            growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(
            bn_size * growth_rate, growth_rate,
            kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=stride, stride=stride))


class DenseNet(nn.Module):
    """Densely connected network.

    Reference:
        Huang et al. Densely Connected Convolutional Networks. CVPR 2017.

    Public keys:
        - ``densenet121``: DenseNet121.
        - ``densenet169``: DenseNet169.
        - ``densenet201``: DenseNet201.
        - ``densenet161``: DenseNet161.
        - ``densenet121_fc512``: DenseNet121 + FC.
    """
    def __init__(self, num_classes, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, fc_dims=None, dropout_p=None, last_stride=2, **kwargs):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features

        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:

                if i == len(block_config) - 2:
                    stride = last_stride
                else:
                    stride = 2

                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=stride)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = num_features
        self.orig_feature_dim = num_features
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_p)

        # Linear layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params(self.fc)
        self._init_params(self.classifier)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer.

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), 'fc_dims must be either list or tuple, but got {}'.format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self, x):
        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


class DensenetABD(nn.Module):

    def __init__(self, num_classes,
                 backbone,
                 *,
                 fd_config=None,
                 attention_config=None,
                 dropout_optimizer=None,
                 fc_dims=(),
                 **kwargs):

        super().__init__()

        self.backbone1 = backbone.features[:6]
        self.backbone2 = backbone.features[6:-2]
        self.backbone3_1 = deepcopy(backbone.features[-2:])
        self.backbone3_2 = deepcopy(backbone.features[-2:])

        # Begin Feature Distilation
        if fd_config is None:
            fd_config = {'parts': (), 'use_conv_head': False}
        from .tricks.feature_distilation import FeatureDistilationTrick
        self.feature_distilation = FeatureDistilationTrick(
            fd_config['parts'],
            channels={'a': [], 'b': [], 'c': list(range(128))},
            use_conv_head=fd_config['use_conv_head']
        )
        backbone._init_params(self.feature_distilation)
        self.dummy_fd = DummyFD(lambda: self.feature_distilation)
        # End Feature Distilation

        self.global_avgpool = backbone.global_avgpool

        output_dim = backbone.orig_feature_dim
        feature_dim = fc_dims[0]

        self.global_reduction = nn.Sequential(
            nn.Linear(output_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            *([dropout_optimizer] if dropout_optimizer is not None else [])
        )

        self.global_classifier = nn.Linear(feature_dim, num_classes)
        backbone._init_params(self.global_reduction)
        backbone._init_params(self.global_classifier)

        self.dim = feature_dim

        self.abd_reduction = nn.Sequential(
            nn.Conv2d(output_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            *([dropout_optimizer] if dropout_optimizer is not None else [])
        )
        backbone._init_params(self.abd_reduction)
        self.get_attention_module()
        self.attention_config = attention_config

        try:
            part_num = int(os.environ.get('part_num'))
        except (TypeError, ValueError):
            part_num = 2

        self.part_num = part_num

        for i in range(1, part_num + 1):
            c = nn.Linear(feature_dim, num_classes)
            setattr(self, f'classifier_p{i}', c)
            self._init_params(c)

    def backbone_convs(self):

        convs = [
            self.backbone1,
            self.backbone2,
            self.backbone3_1,
            self.backbone3_2,
        ]

        return convs

    def _init_params(self, x):
        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_attention_module(self):

        from .tricks.attention import DANetHead, CAM_Module, PAM_Module

        in_channels = self.dim
        out_channels = self.dim

        self.before_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, lambda _: lambda x: x)
        self.pam_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, PAM_Module)
        self.cam_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, CAM_Module)
        self.sum_conv1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(out_channels, out_channels, 1))

        self._init_params(self.before_module1)
        self._init_params(self.cam_module1)
        self._init_params(self.pam_module1)
        self._init_params(self.sum_conv1)

    def forward(self, x):

        x = self.backbone1(x)
        x = self.dummy_fd(x)
        layer5 = x
        x = self.backbone2(x)

        predict, xent, triplet = [], [], []

        x1 = self.backbone3_1(x)
        x1 = F.relu(x1)
        x1 = self.global_avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        x1 = self.global_reduction(x1)
        predict.append(x1)
        triplet.append(x1)
        x1 = self.global_classifier(x1)
        xent.append(x1)

        x2 = self.backbone3_2(x)
        x2 = F.relu(x2)
        x2 = self.abd_reduction(x2)

        feature_dict = {
            'cam': (),
            'pam': (),
            'before': (),
            'after': (),
            'layer5': layer5
        }

        margin = x2.size(2) // self.part_num

        for p in range(1, self.part_num + 1):

            f = x2[:, :, margin * (p - 1):margin * p, :]

            if not self.attention_config['parts']:
                # f_after1 = f_before1 = self.before_module1(f)
                f_after1 = f_before1 = f
                feature_dict['before'] = (*feature_dict['before'], f_before1)
                feature_dict['after'] = (*feature_dict['after'], f_after1)
            else:
                f_before1 = self.before_module1(f)
                f_cam1 = self.cam_module1(f)
                f_pam1 = self.pam_module1(f)

                f_sum1 = f_cam1 + f_pam1 + f_before1
                f_after1 = self.sum_conv1(f_sum1)

                feature_dict['cam'] = (*feature_dict['cam'], f_cam1)
                feature_dict['pam'] = (*feature_dict['pam'], f_pam1)
                feature_dict['before'] = (*feature_dict['before'], f_before1)
                feature_dict['after'] = (*feature_dict['after'], f_after1)

            v = self.global_avgpool(f_after1)
            v = v.view(v.size(0), -1)
            triplet.append(v)
            predict.append(v)
            v = getattr(self, f'classifier_p{p}')(v)
            xent.append(v)

        if not self.training and os.environ.get('fake') is None:
            return torch.cat(predict, 1)

        return None, tuple(xent), tuple(triplet), feature_dict


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)

    # '.'s are no longer allowed in module names, but pervious _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(pretrain_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            pretrain_dict[new_key] = pretrain_dict[key]
            del pretrain_dict[key]

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


"""
Dense network configurations:
--
densenet121: num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
densenet169: num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)
densenet201: num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)
densenet161: num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24)
"""


def densenet121(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def densenet169(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet169'])
    return model


def densenet201(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet201'])
    return model


def densenet161(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet161'])
    return model


def densenet121_fc512(num_classes, loss='softmax', pretrained=True, **kwargs):
    model = DenseNet(
        num_classes=num_classes,
        loss=loss,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained:
        init_pretrained_weights(model, model_urls['densenet121'])
    return model


def make_function_abd(name, config, base_class=DenseNet, url=model_urls['densenet121'], last_stride=2):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        backbone = base_class(num_classes, fc_dims=config['fc_dims'])
        init_pretrained_weights(backbone, url)
        return DensenetABD(
            num_classes=num_classes,
            backbone=backbone,
            **config,
            **kwargs
        )

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


from collections import OrderedDict

configurations = OrderedDict([
    (
        'fc_dims',
        [
            (None, '_nofc'),
            ([256], '_fc256'),
            ([512], '_fc512'),
            ([1024], '_fc1024'),
        ],
    ),
    (
        'fd_config',
        [
            (
                {
                    'parts': parts,
                    'use_conv_head': use_conv_head
                },
                f'_fd_{parts_name}_{"head" if use_conv_head else "nohead"}'
            )
            for parts, parts_name in [
                [('ab', 'c'), 'ab_c'],
                [('ab',), 'ab'],
                [('a',), 'a'],
                [(), 'none'],
                [('abc',), 'abc']
            ]
            for use_conv_head in (True, False)
        ],
    ),
    (
        'attention_config',
        [
            (
                {
                    'parts': parts,
                    'use_conv_head': use_conv_head
                },
                f'_dan_{parts_name}_{"head" if use_conv_head else "nohead"}'
            )
            for parts, parts_name in [
                [('cam', 'pam'), 'cam_pam'],
                [('cam',), 'cam'],
                [('pam',), 'pam'],
                [(), 'none'],
            ]
            for use_conv_head in (True, False)
        ]
    )
])

import itertools

fragments = list(itertools.product(*configurations.values()))
keys = list(configurations.keys())

name_function_mapping = {}

for fragment in fragments:

    name = 'densenet121'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_abd(name, config)

for fragment in fragments:

    name = 'densenet121_ls1'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_abd(name, config, last_stride=1)
