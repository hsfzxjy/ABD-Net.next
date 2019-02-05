from __future__ import absolute_import
from __future__ import division

import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet50', 'resnet50_fc512']

channels = {
    'a': [4, 40, 64, 68, 70, 71, 101, 102, 127, 141, 152, 158, 162, 164, 171, 172, 175, 186, 201, 209, 225, 227, 246],
    'b': [2, 11, 12, 17, 23, 24, 28, 30, 36, 39, 47, 48, 49, 57, 59, 60, 61, 66, 77, 78, 83, 87, 88, 89, 91, 93, 99, 107, 110, 117, 120, 121, 123, 124, 126, 133, 139, 140, 145, 151, 155, 165, 168, 174, 180, 185, 192, 199, 202, 211, 216, 217, 219, 222, 230, 239, 245, 247, 249, 251],
    'c': [0, 1, 3, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20, 21, 22, 25, 26, 27, 29, 31, 32, 33, 34, 35, 37, 38, 41, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 56, 58, 62, 63, 65, 67, 69, 72, 73, 74, 75, 76, 79, 80, 81, 82, 84, 85, 86, 90, 92, 94, 95, 96, 97, 98, 100, 103, 104, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 118, 119, 122, 125, 128, 129, 130, 131, 132, 134, 135, 136, 137, 138, 142, 143, 144, 146, 147, 148, 149, 150, 153, 154, 156, 157, 159, 160, 161, 163, 166, 167, 169, 170, 173, 176, 177, 178, 179, 181, 182, 183, 184, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 200, 203, 204, 205, 206, 207, 208, 210, 212, 213, 214, 215, 218, 220, 221, 223, 224, 226, 228, 229, 231, 232, 233, 234, 235, 236, 237, 238, 240, 241, 242, 243, 244, 248, 250, 252, 253, 254, 255],

}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes, loss, block, layers,
                 last_stride=2,
                 fc_dims=None,
                 dropout_p=None,
                 *,
                 fd_config=None,
                 attention_config=None,
                 dropout_optimizer=None,
                 sum_fusion: bool=False,
                 tricky: int=0,
                 **kwargs):

        self.sum_fusion = sum_fusion
        self.tricky = tricky

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

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

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 512 * block.expansion
        # Begin Attention Module
        if attention_config is None:
            attention_config = {'parts': (), 'use_conv_head': False}
        from .tricks.attention import AttentionModule

        self.attention_module = AttentionModule(
            attention_config['parts'],
            num_features,
            use_conv_head=attention_config['use_conv_head'],
            sum_fusion=self.sum_fusion
        )
        if not sum_fusion:
            self.feature_dim = num_features = num_features + self.attention_module.output_dim
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from .tricks.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        if self.tricky == 1:
            self.classifier2 = nn.Linear(num_features, num_classes)

        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def backbone_convs(self):

        return [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward_feature_distilation(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        all_layers = [x]

        layer5 = x

        B, C, H, W = x.shape

        for cs, cam in self.feature_distilation.cam_modules:
            c_tensor = torch.tensor(cs).cuda()

            new_x = x[:, c_tensor]
            new_x = cam(new_x)
            x[:, c_tensor] = new_x

        x = self.layer2(x)
        all_layers.append(x)
        x = self.layer3(x)
        all_layers.append(x)
        x = self.layer4(x)
        all_layers.append(x)
        return x, layer5, tuple(all_layers)

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

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        f, layer5, all_layers = self.forward_feature_distilation(x)

        feature_dict, pooling = self.attention_module(f)

        if not self.sum_fusion:
            attention_parts = []
            for k in feature_dict:
                pool = pooling[k]
                _f = pool(feature_dict[k])
                attention_parts.append(_f.view(_f.size(0), -1))

            feature_dict['before'] = f
            v = self.global_avgpool(f)
            v = v.view(v.size(0), -1)

            v = torch.cat([v, *attention_parts], 1)
        else:
            feature_dict['before'] = f
            f = sum(feature_dict.values())
            feature_dict['after'] = f
            v = self.global_avgpool(f)
            v = v.view(v.size(0), -1)

        feature_dict['layer5'] = layer5
        feature_dict['all_layers'] = all_layers

        if self.tricky == 2:
            triplet_feature = self.global_avgpool(feature_dict['before'])
            triplet_feature = triplet_feature.view(triplet_feature.size(0), -1)
        else:
            triplet_feature = v

        v_before_fc = v
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            if self.tricky == 1:
                return torch.cat([v, v_before_fc], 1)
            if os.environ.get('NOFC'):
                return v_before_fc
            else:
                return v

        y = self.classifier(v)

        if self.tricky == 1:
            y = (y, self.classifier2(v_before_fc))

        return f, y, v, feature_dict

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def resnet50(num_classes, loss, pretrained='imagenet', **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=2,
        fc_dims=None,
        dropout_p=None,
        **kwargs
    )
    if pretrained == 'imagenet':
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

#


def make_function_50(name, config):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        model = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            dropout_p=None,
            **config,
            **kwargs
        )
        if pretrained == 'imagenet':
            init_pretrained_weights(model, model_urls['resnet50'])
        return model

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


def make_function_sf_50(name, config):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        model = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            dropout_p=None,
            sum_fusion=True,
            **config,
            **kwargs
        )
        if pretrained == 'imagenet':
            init_pretrained_weights(model, model_urls['resnet50'])
        return model

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


def make_function_sf_ls1_50(name, config):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        model = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=1,
            dropout_p=None,
            sum_fusion=True,
            **config,
            **kwargs
        )
        if pretrained == 'imagenet':
            init_pretrained_weights(model, model_urls['resnet50'])
        return model

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


def make_function_sf_tricky_50(name, config, tricky):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        model = ResNet(
            num_classes=num_classes,
            loss=loss,
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            last_stride=2,
            dropout_p=None,
            sum_fusion=True,
            tricky=tricky,
            **config,
            **kwargs
        )
        if pretrained == 'imagenet':
            init_pretrained_weights(model, model_urls['resnet50'])
        return model

    _func.config = config

    name_function_mapping[name] = _func
    globals()[name] = _func


def resnet50_fc512(num_classes, loss, pretrained='imagenet', **kwargs):
    model = ResNet(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=1,
        fc_dims=[512],
        dropout_p=None,
        **kwargs
    )
    if pretrained == 'imagenet':
        init_pretrained_weights(model, model_urls['resnet50'])
    return model


from collections import OrderedDict

configurations = OrderedDict([
    (
        'fc_dims',
        [
            (None, '_nofc'),
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

    name = 'resnet50'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_50(name, config)

for fragment in fragments:

    name = 'resnet50_sf'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_50(name, config)


#
for fragment in fragments:

    name = 'resnet50_sf_ls1'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_ls1_50(name, config)


for fragment in fragments:

    name = 'resnet50_sf_tr1'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 1)

for fragment in fragments:

    name = 'resnet50_sf_tr2'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 2)
