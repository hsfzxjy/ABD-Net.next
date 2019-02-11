###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import torch
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'get_attention_module_instance', 'AttentionModule']


class AttentionModule(nn.Module):

    def __init__(
        self,
        module_names: 'subset of ("pam", "cam")',
        dim: int,
        *,
        use_conv_head: bool=False,
        use_avg_pool: bool=True,
        sum_fusion: bool=False
    ):
        super().__init__()
        print(module_names)
        self.modules_list = []
        for name in module_names:
            module = get_attention_module_instance(name, dim, use_conv_head=use_conv_head, sum_fusion=sum_fusion)
            setattr(self, f'_{name}_module', module)  # force gpu

            if use_avg_pool:
                pool = nn.AdaptiveAvgPool2d(1)
                setattr(self, f'_{name}_avgpool', pool)  # force gpu
            else:
                pool = None

            self.modules_list.append((name, module, pool))

        self.output_dim = len(self.modules_list) * dim if not sum_fusion else dim

    def forward(self, x):

        xs = {}
        pooling = {}
        for name, module, pool in self.modules_list:
            f = module(x)
            xs[name] = f
            pooling[name] = pool
            # xs.append(f.view(f.size(0), -1))
        return xs, pooling


def get_attention_module_instance(
    name: 'cam|pam',
    dim: int,
    *,
    use_conv_head: bool=False,  # DEPRECATED
    sum_fusion: bool=True
):

    name = name.lower()
    assert name in ('cam', 'pam')

    module_class = {'cam': CAM_Module, 'pam': PAM_Module}[name]

    use_conv_head = not sum_fusion and name == 'cam'

    if use_conv_head:
        return DANetHead(dim, dim, nn.BatchNorm2d, module_class)
    else:
        return module_class(dim)


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, module_class):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4

        self.conv5c = nn.Sequential(
            nn.Conv2d(
                in_channels,
                inter_channels,
                3,
                padding=1,
                bias=False
            ),
            norm_layer(inter_channels),
            nn.ReLU()
        )

        self.attention_module = module_class(inter_channels)
        self.conv52 = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                inter_channels,
                3,
                padding=1,
                bias=False
            ),
            norm_layer(inter_channels),
            nn.ReLU()
        )

        self.conv7 = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1)
        )

    def forward(self, x):

        feat2 = self.conv5c(x)
        sc_feat = self.attention_module(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        return sc_output


class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out  # , attention_mask


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.tensor(torch.max(energy, -1, keepdim=True)[0], device='cuda:5').expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
