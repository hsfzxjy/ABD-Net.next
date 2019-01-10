from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict
import math
import re
import os
import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F
import torchvision


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

channels = {
    'a': [
        7, 9, 20, 24, 28, 36, 38, 42, 44, 52, 61, 63, 66, 74, 77, 80, 89, 105, 108, 109, 119, 120
    ],

    'b': [2, 6, 14, 17, 23, 29, 30, 33, 34, 35, 39, 43, 48, 51, 62, 72, 81, 92, 101, 103, 115, 123, 127],
    'c': [0, 1, 3, 4, 5, 8, 10, 11, 12, 13, 15, 16, 18, 19, 21, 22, 25, 26, 27, 31, 32, 37, 40, 41, 45, 46, 47, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 67, 68, 69, 70, 71, 73, 75, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 102, 104, 106, 107, 110, 111, 112, 113, 114, 116, 117, 118, 121, 122, 124, 125, 126]
}


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
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
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densely connected network

    Reference:
    Huang et al. Densely Connected Convolutional Networks. CVPR 2017.
    """

    def __init__(
            self,
            num_classes,
            loss,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            num_init_features=64,
            bn_size=4,
            drop_rate=0,
            fc_dims=None,
            dropout_p=None,
            *,
            fd_config=None,
            attention_config=None,
            dropout_optimizer=None,
            **kwargs):

        super(DenseNet, self).__init__()
        self.loss = loss

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
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # Begin Feature Distilation
        if fd_config is None:
            fd_config = {'parts': (), 'use_conv_head': False}
        from .tricks.feature_distilation import FeatureDistilationTrick
        self.feature_distilation = FeatureDistilationTrick(
            fd_config['parts'],
            channels=channels,
            use_conv_head=fd_config['use_conv_head']
        )
        # End Feature Distilation

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Begin Attention Module
        if attention_config is None:
            attention_config = {'parts': (), 'use_conv_head': False}
        from .tricks.attention import AttentionModule

        self.attention_module = AttentionModule(
            attention_config['parts'],
            num_features,
            use_conv_head=attention_config['use_conv_head']
        )
        self.feature_dim = num_features = num_features + self.attention_module.output_dim
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from .tricks.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)

        # Linear layer
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def forward_feature_distilation(self, x):

        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 5:
                B, C, H, W = x.shape

                for cs, cam in self.feature_distilation.cam_modules:
                    c_tensor = torch.tensor(cs).cuda()

                    new_x = x[:, c_tensor]
                    new_x = cam(new_x)
                    x[:, c_tensor] = new_x

        return x

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_optimizer):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(dropout_optimizer)
            # if dropout_p is not None:
            #     layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
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
        f = self.forward_feature_distilation(x)

        feature_dict = self.attention_module(f)
        attention_parts = [
            part.view(part.size(0), -1) for part in
            feature_dict.values()
        ]
        feature_dict['before'] = f

        f = F.relu(f, inplace=False)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        v = torch.cat([v, *attention_parts], 1)

        v_before_fc = v
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            if os.environ.get('NOFC'):
                return v_before_fc
            else:
                return v

        y = self.classifier(v)

        return f, y, v, feature_dict

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, v
        else:
            return f, y, v
            # raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
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
    print("Initialized model with pretrained weights from {}".format(model_url))


"""
Dense network configurations:
--
densenet121: num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16)
densenet169: num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32)
densenet201: num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32)
densenet161: num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24)
"""


def make_function_121(name, config):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        model = DenseNet(
            num_classes=num_classes,
            loss=loss,
            num_init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            dropout_p=None,
            **config,
            **kwargs
        )
        if pretrained == 'imagenet':
            init_pretrained_weights(model, model_urls['densenet121'])
        return model

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


def make_function_161(name, config):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):

        model = DenseNet(
            num_classes=num_classes,
            loss=loss,
            num_init_features=96,
            growth_rate=48,
            block_config=(6, 12, 36, 24),
            dropout_p=None,
            **config,
            **kwargs
        )
        if pretrained == 'imagenet':
            init_pretrained_weights(model, model_urls['densenet161'])
        return model

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


configurations = OrderedDict([
    (
        'fc_dims',
        [
            (None, '_nofc'),
            ([512], '_fc512'),
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
                [(), 'none']
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

configurations: 'Dict[str, List[Tuple[config, name_frag]]]'

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

    make_function_121(name, config)

for fragment in fragments:

    name = 'densenet161'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_161(name, config)
