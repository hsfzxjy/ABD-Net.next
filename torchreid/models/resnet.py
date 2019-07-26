from __future__ import absolute_import
from __future__ import division

import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from copy import deepcopy

from torchreid.components import branches
from torchreid.components.shallow_cam import ShallowCAM

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

def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print('Initialized model with pretrained weights from {}'.format(model_url))


class ResNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3,
        )

    def backbone_modules(self):

        return [self.backbone1, self.backbone2]

    def forward(self, x):

        x = self.backbone1(x)
        before_intermediate = x
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, {'intermediate': (intermediate,), 'before_intermediate': before_intermediate}

class ResNetDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone = deepcopy(backbone.layer4)

        self.out_dim = 2048

    def backbone_modules(self):

        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)

class ResNetMGNLikeCommonBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(args, 256)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3[0],
        )

    def backbone_modules(self):

        return [self.backbone1, self.backbone2]

    def forward(self, x):

        x = self.backbone1(x)
        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate

class ResNetMGNLikeDeepBranch(nn.Module):

    def __init__(self, owner, backbone, args):

        super().__init__()

        self.backbone = nn.Sequential(
            *deepcopy(backbone.layer3[1:]),
            deepcopy(backbone.layer4)
        )
        self.out_dim = 2048

    def backbone_modules(self):

        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)


class MultiBranchResNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):

        return ResNetCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return ResNetDeepBranch(self, backbone, args)

class MultiBranchMGNLikeResNet(branches.MultiBranchNetwork):

    def _get_common_branch(self, backbone, args):

        return ResNetMGNLikeCommonBranch(self, backbone, args)

    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return ResNetMGNLikeDeepBranch(self, backbone, args)


class ResNetOld(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 dropout_p=None,
                 *,
                 fd_config={'parts': (), 'use_conv_head': False},
                 attention_config={'parts': ('cam', 'pam')},
                 dropout_optimizer=None,
                 fc_dims=(1024,),
                 **kwargs):

        self.sum_fusion = True
        last_stride = 1

        super().__init__()
        self.feature_dim = 2048  # 512 * block.expansion

        # backbone network
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], last_stride)
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

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        self.get_attention_module()
        self.attention_config = attention_config
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from torchreid.components.dropout import SimpleDropoutOptimizer
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

        from torchreid.components.attention import DANetHead, CAM_Module, PAM_Module

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

    def backbone_modules(self):

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
            'intermediate': layer5
        }

        margin = x2.size(2) // self.part_num

        for p in range(1, self.part_num + 1):

            f = x2[:, :, margin * (p - 1):margin * p, :]

            if not self.attention_config['parts']:
                # f_after1 = f_before1 = self.before_module1(f)
                f_after1 = f_before1 = F.relu(f)
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

        return torch.cat(predict_features, 1), tuple(xent_features), tuple(triplet_features), feature_dict


class ResNetNp2(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 dropout_p=None,
                 *,
                 fd_config={'parts': (), 'use_conv_head': False},
                 attention_config={'parts': ()},
                 dropout_optimizer=None,
                 fc_dims=(512,),
                 **kwargs):

        self.sum_fusion = True
        last_stride = 1

        super().__init__()
        self.feature_dim = 2048  # 512 * block.expansion

        # backbone network
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], last_stride)
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

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        self.get_attention_module()
        self.attention_config = attention_config
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from torchreid.components.dropout import SimpleDropoutOptimizer
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

        from torchreid.components.attention import DANetHead, CAM_Module, PAM_Module

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

    def backbone_modules(self):

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
            'intermediate': layer5
        }

        margin = x2.size(2) // self.part_num

        for p in range(1, self.part_num + 1):

            f = x2[:, :, margin * (p - 1):margin * p, :]

            if not self.attention_config['parts']:
                # f_after1 = f_before1 = self.before_module1(f)
                f_after1 = f_before1 = F.relu(f)
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

        return torch.cat(predict_features, 1), tuple(xent_features), tuple(triplet_features), feature_dict

def resnet50_backbone(args):

    network = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=args['resnet_last_stride'],  # Always remove down-sampling
    )
    init_pretrained_weights(network, model_urls['resnet50'])

    return network


def resnet50(num_classes, args, **kw):

    backbone = resnet50_backbone(args)
    return MultiBranchResNet(backbone, args, num_classes)

def resnet50_mgn_like(num_classes, args, **kw):

    backbone = resnet50_backbone(args)
    return MultiBranchMGNLikeResNet(backbone, args, num_classes)

def resnet50_abd_old(num_classes, args, **kw):

    return ResNetOld(num_classes, 0.5)

def resnet50_abd_old_baseline(num_classes, args, **kw):

    return ResNetOld(num_classes, 0.5, attention_config={'parts': ()})

def resnet50_abd_np2(num_classes, args, **kw):

    return ResNetNp2(num_classes, 0.5)
