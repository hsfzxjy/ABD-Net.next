from __future__ import absolute_import
from __future__ import division

import os
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from copy import deepcopy

import logging

logger = logging.getLogger(__name__)


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


class DummySum(nn.Module):

    def forward(self, x, y, z):

        return x + y + z


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


class ResNetBackbone(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

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


class ResNetABD(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 dropout_p=None,
                 *,
                 fd_config=None,
                 attention_config=None,
                 dropout_optimizer=None,
                 fc_dims=(),
                 **kwargs):

        self.sum_fusion = True
        last_stride = 1

        super(ResNetABD, self).__init__()
        self.feature_dim = 2048  # 512 * block.expansion

        # backbone network
        backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3], last_stride)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer4_normal_branch = deepcopy(self.layer4)

        dim = fc_dims[0]
        self.dim = dim

        # Begin Feature Distilation
        if fd_config is None:
            fd_config = {'parts': (), 'use_conv_head': False}
        from .tricks.feature_distilation import FeatureDistilationTrick
        self.feature_distilation = FeatureDistilationTrick(
            fd_config['parts'],
            channels=channels,
            use_conv_head=fd_config['use_conv_head']
        )
        self._init_params(self.feature_distilation)
        self.dummy_fd = DummyFD(lambda: self.feature_distilation)
        # End Feature Distilation

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        self.get_attention_module()
        self.attention_config = attention_config
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from .tricks.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        if os.environ.get('dropout_reduction'):
            dropout = [dropout_optimizer]
        else:
            dropout = []

        # fc_dims = [1024]
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.reduction_tr = nn.Sequential(
            nn.Conv2d(2048, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            *dropout
        )
        self._init_params(self.reduction_tr)

        try:
            part_num = int(os.environ.get('part_num'))
        except (TypeError, ValueError):
            part_num = 2

        self.part_num = part_num

        for i in range(1, part_num + 1):
            c = nn.Linear(dim, num_classes)
            setattr(self, f'classifier_p{i}', c)
            self._init_params(c)

        self._init_params(self.fc)
        self._init_params(self.classifier)

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

    def backbone_convs(self):

        convs = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer4_normal_branch,
        ]

        return convs

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

    def _init_params(self, x):
        if x is None:
            return

        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x = self.dummy_fd(x)
        layer5 = x
        x = self.layer2(x)
        x = self.layer3(x)

        triplet_features = []
        xent_features = []
        predict_features = []

        # normal branch
        x1 = x
        x1 = self.layer4_normal_branch(x1)
        x1 = self.global_avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        triplet_features.append(x1)
        x1 = self.fc(x1)
        predict_features.append(x1)
        x1 = self.classifier(x1)
        xent_features.append(x1)

        # our branch
        x2 = x
        x2 = self.layer4(x2)
        x2 = self.reduction_tr(x2)

        feature_dict = {
            'cam': [],
            'pam': [],
            'before': [],
            'after': [],
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
            triplet_features.append(v)
            predict_features.append(v)
            v = getattr(self, f'classifier_p{p}')(v)
            xent_features.append(v)

        if not self.training and os.environ.get('fake') is None:
            return torch.cat(predict_features, 1)

        return None, tuple(xent_features), tuple(triplet_features), feature_dict


class ResNetABDConcat(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 dropout_p=None,
                 *,
                 fd_config=None,
                 attention_config=None,
                 dropout_optimizer=None,
                 fc_dims=(),
                 **kwargs):

        self.sum_fusion = True
        last_stride = 1

        super().__init__()
        self.feature_dim = 2048  # 512 * block.expansion

        # backbone network
        backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3], last_stride)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # self.layer4_normal_branch = nn.Sequential(
        #     Bottleneck(
        #         1024,
        #         512,
        #         stride=normal_branch_stride,
        #         downsample=nn.Sequential(
        #             nn.Conv2d(
        #                 1024, 2048, kernel_size=1, stride=normal_branch_stride, bias=False
        #             ),
        #             nn.BatchNorm2d(2048)
        #         )
        #     ),
        #     Bottleneck(2048, 512),
        #     Bottleneck(2048, 512)
        # )
        # self.layer4_normal_branch.load_state_dict(backbone.layer4.state_dict())
        #
        self.layer4_normal_branch = deepcopy(self.layer4)

        dim = fc_dims[0]
        self.dim = dim

        # Begin Feature Distilation
        if fd_config is None:
            fd_config = {'parts': (), 'use_conv_head': False}
        from .tricks.feature_distilation import FeatureDistilationTrick
        self.feature_distilation = FeatureDistilationTrick(
            fd_config['parts'],
            channels=channels,
            use_conv_head=fd_config['use_conv_head']
        )
        self._init_params(self.feature_distilation)
        self.dummy_fd = DummyFD(lambda: self.feature_distilation)
        # End Feature Distilation

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        self.get_attention_module()
        self.attention_config = attention_config
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from .tricks.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        if os.environ.get('dropout_reduction'):
            dropout = [dropout_optimizer]
        else:
            dropout = []

        # fc_dims = [1024]
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)
        # self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.reduction_tr = nn.Sequential(
            nn.Conv2d(2048, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            *dropout
        )
        self._init_params(self.reduction_tr)

        try:
            part_num = int(os.environ.get('part_num'))
        except (TypeError, ValueError):
            part_num = 2

        self.part_num = part_num

        self.classifier = nn.Linear(dim * (part_num + 1), num_classes)

        # for i in range(1, part_num + 1):
        #     c = nn.Linear(dim, num_classes)
        #     setattr(self, f'classifier_p{i}', c)
        #     self._init_params(c)

        self._init_params(self.fc)
        self._init_params(self.classifier)

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

    def backbone_convs(self):

        convs = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer4_normal_branch,
        ]

        return convs

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

    def _init_params(self, x):
        if x is None:
            return

        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # B, C, H, W = x.shape

        # for cs, cam in self.feature_distilation.cam_modules:
        #     c_tensor = torch.tensor(cs).cuda()

        #     new_x = x[:, c_tensor]
        #     new_x = cam(new_x)
        #     x[:, c_tensor] = new_x

        # layer5 = x

        x = self.dummy_fd(x)
        layer5 = x
        x = self.layer2(x)
        x = self.layer3(x)

        triplet_features = []
        xent_features = []
        predict_features = []

        # normal branch
        x1 = x
        x1 = self.layer4_normal_branch(x1)
        x1 = self.global_avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        # triplet_features.append(x1)
        x1 = self.fc(x1)
        predict_features.append(x1)
        # x1 = self.classifier(x1)
        # xent_features.append(x1)

        # our branch
        x2 = x
        x2 = self.layer4(x2)
        x2 = self.reduction_tr(x2)

        feature_dict = {
            'cam': [],
            'pam': [],
            'before': [],
            'after': [],
            'layer5': layer5
        }

        margin = x.size(2) // self.part_num

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
            # triplet_features.append(v)
            predict_features.append(v)
            # v = getattr(self, f'classifier_p{p}')(v)
            # xent_features.append(v)

        if not self.training and os.environ.get('fake') is None:
            return torch.cat(predict_features, 1)

        xent_features.append(self.classifier(torch.cat(predict_features, 1)))
        triplet_features.append(torch.cat(predict_features, 1))

        return None, tuple(xent_features), tuple(triplet_features), feature_dict

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


def make_function_sf_tricky_50(name, config, tricky, base_class):

    def _func(num_classes, loss, pretrained='imagenet', **kwargs):
        print(config)
        return base_class(
            num_classes=num_classes,
            last_stride=2,
            dropout_p=None,
            sum_fusion=True,
            tricky=tricky,
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

configurations: 'Dict[str, List[Tuple[config, name_frag]]]'

import itertools

fragments = list(itertools.product(*configurations.values()))
keys = list(configurations.keys())

name_function_mapping = {}


for fragment in fragments:

    name = f'resnet50_sf_abd'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 12, base_class=ResNetABD)

for fragment in fragments:

    name = f'resnet50_sf_cam'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 12, base_class=ResNetCAM)

for fragment in fragments:

    name = f'resnet50_sf_pam'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 12, base_class=ResNetPAM)

for fragment in fragments:

    name = f'resnet50_abl'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 12, base_class=ResNetAblation)

for fragment in fragments:

    name = f'resnet50_abd_concat'
    config = {}
    for key, (sub_config, name_frag) in zip(keys, fragment):
        name += name_frag
        config.update({key: sub_config})

    make_function_sf_tricky_50(name, config, 12, base_class=ResNetABDConcat)
