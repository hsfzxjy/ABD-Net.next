import os
import torch
from torch import nn
from torch.nn import functional as F

from . import densenet as densenet_
from .attention import PAM_Module, CAM_Module

__all__ = ['densenet121_CAM_cl_cat_fc512', 'densenet121_CAM_noncl_cat_fc512', 'densenet161_CAM_noncl_cat_fc512', 'densenet201_CAM_noncl_cat_fc512', 'densenet161_CAM_noncl_cat_trick_fc512', 'densenet161_CAM_noncl_cat_trick_1_4_fc512', 'densenet161_CAM_noncl_cat_1_4_fc512']

channels = sorted([5,
                   98,
                   228,
                   262,
                   377,
                   507,
                   524,
                   531,
                   535,
                   546,
                   549,
                   554,
                   564,
                   570,
                   579,
                   585,
                   586,
                   589,
                   590,
                   599,
                   602,
                   603,
                   608,
                   609,
                   616,
                   627,
                   628,
                   629,
                   643,
                   645,
                   648,
                   660,
                   665,
                   668,
                   675,
                   678,
                   686,
                   691,
                   697,
                   702,
                   706,
                   710,
                   711,
                   713,
                   719,
                   723,
                   726,
                   729,
                   732,
                   734,
                   738,
                   740,
                   745,
                   747,
                   752,
                   762,
                   779,
                   783,
                   784,
                   791,
                   798,
                   801,
                   802,
                   810,
                   811,
                   812,
                   814,
                   824,
                   829,
                   832,
                   837,
                   844,
                   850,
                   857,
                   858,
                   862,
                   863,
                   867,
                   868,
                   878,
                   879,
                   889,
                   911,
                   913,
                   917,
                   919,
                   926,
                   927,
                   930,
                   931,
                   938,
                   942,
                   950,
                   954,
                   955,
                   956,
                   957,
                   958,
                   962,
                   963,
                   969,
                   972,
                   975,
                   976,
                   977,
                   982,
                   985,
                   987,
                   988,
                   995,
                   1002,
                   1005,
                   1008,
                   1012,
                   1013,
                   1014,
                   1016,
                   1023,
                   100,
                   152,
                   162,
                   518,
                   520,
                   541,
                   544,
                   547,
                   555,
                   558,
                   568,
                   572,
                   583,
                   594,
                   597,
                   598,
                   604,
                   615,
                   642,
                   685,
                   696,
                   698,
                   720,
                   731,
                   733,
                   750,
                   770,
                   786,
                   799,
                   818,
                   819,
                   827,
                   841,
                   854,
                   855,
                   884,
                   887,
                   890,
                   899,
                   907,
                   914,
                   932,
                   935,
                   944,
                   949,
                   951,
                   959,
                   960,
                   961,
                   973,
                   978,
                   986,
                   992,
                   1007,
                   1010,
                   1017,
                   532,
                   605,
                   638,
                   744,
                   754,
                   782,
                   789,
                   800,
                   836,
                   860,
                   869,
                   901,
                   983,
                   464,
                   542,
                   714,
                   765,
                   806,
                   817,
                   856,
                   898,
                   916,
                   974,
                   980,
                   1015,
                   707,
                   838,
                   894,
                   947,
                   620,
                   656,
                   578,
                   ])
b_channels = sorted(list(set(range(1023)) - set(channels)))


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, type_):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        # inter_channels = in_channels
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.type_ = type_.lower()
        assert self.type_ in 'cp'

    def forward(self, x):

        if self.type_ == 'p':
            feat1 = self.conv5a(x)
            sa_feat, _ = self.sa(feat1)
            sa_conv = self.conv51(sa_feat)
            sa_output = self.conv6(sa_conv)
            return sa_output
        else:
            feat2 = self.conv5c(x)
            sc_feat = self.sc(feat2)
            sc_conv = self.conv52(sc_feat)
            sc_output = self.conv7(sc_conv)
            return sc_output
        # feat_sum = sa_conv + sc_conv

        # sasc_output = self.conv8(feat_sum)

        # output = [sasc_output]
        # output.append(sa_output)
        # output.append(sc_output)
        # return tuple(output)


class DensenetCAMCat(densenet_.DenseNet):

    def __init__(self, num_classes, loss, fc_dims, cluster=True, pam_division=1, **kwargs):

        kw = dict(num_classes=num_classes,
                  loss=loss,
                  num_init_features=64,
                  growth_rate=32,
                  block_config=(6, 12, 24, 16),
                  fc_dims=None,
                  dropout_p=None,)
        kw.update(kwargs)

        super().__init__(
            **kw
        )

        self.cluster = cluster
        self.ca1 = DANetHead(len(channels), len(channels), nn.BatchNorm2d, type_='c')
        self.ca2 = DANetHead(len(b_channels), len(b_channels), nn.BatchNorm2d, type_='c')
        self.ca = DANetHead(self.feature_dim, self.feature_dim, nn.BatchNorm2d, type_='c')
        self.pa = DANetHead(self.feature_dim, self.feature_dim // pam_division, nn.BatchNorm2d, type_='p')
        print(self.pa)
        self.fc = self._construct_fc_layer(fc_dims, self.feature_dim * 2 + self.feature_dim // pam_division, dropout_p=None)
        print(self.fc)
        # feature_dim changed after _construct_fc_layer, so we must construct
        # classifier again
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.ca_avgpool = nn.AdaptiveAvgPool2d(1)
        self.pa_avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.features(x)
        print('F!', f.size())
        old_f = f

        if self.cluster:

            ca = f
            c_tensor = torch.tensor(channels).to(torch.device('cuda'))
            bc_tensor = torch.tensor(b_channels).to(torch.device('cuda'))

            # x1 = torch.index_select(x, 1, c_tensor)
            # x2 = torch.index_select(x, 1, bc_tensor)
            # print(x.shape)
            x1 = ca[:, c_tensor]
            x2 = ca[:, bc_tensor]

            x1 = self.ca1(x1)
            x2 = self.ca2(x2)

            ca[:, c_tensor] = x1
            ca[:, bc_tensor] = x2

        else:

            ca = f
            ca = self.ca(f)
        ca = self.ca_avgpool(ca)
        ca = ca.view(ca.size(0), -1)

        pa = f
        pa = self.pa(f)
        pa = self.pa_avgpool(pa)
        pa = pa.view(pa.size(0), -1)

        f = F.relu(f)
        v = self.global_avgpool(f)

        v = v.view(v.size(0), -1)

        v = torch.cat((v, pa, ca), 1)
        # v = pa
        #
        if os.environ.get('NOFC') and not self.training:
            return v.view(v.size(0), -1)
        old_v = v
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            if os.environ.get('FCCNFC'):
                print(pa.size(), ca.size(), old_v.size(), v.size())
                return torch.cat((v, old_v), 1)
            return v.view(v.size(0), -1)

        y = self.classifier(v)
        print(self.classifier.weight)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, v
        else:
            return old_f, y, v
            # raise KeyError("Unsupported loss: {}".format(self.loss))


def densenet121_CAM_cl_cat(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=None,
        cluster=True,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet121'])

    return model


def densenet121_CAM_cl_cat_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=True,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet121'])

    return model


#
def densenet121_CAM_noncl_cat(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=None,
        cluster=False,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet121'])

    return model


def densenet121_CAM_noncl_cat_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=False,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet121'])

    return model


def densenet161_CAM_noncl_cat_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=False,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet161'])

    return model


def densenet161_CAM_noncl_cat_trick_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=False,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        dropout_p=0.5,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet161'])

    return model


def densenet161_CAM_noncl_cat_1_4_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=False,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        pam_division=4,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet161'])

    return model


def densenet161_CAM_noncl_cat_trick_1_4_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=False,
        num_init_features=96,
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        dropout_p=0.5,
        pam_division=4,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet161'])

    return model


def densenet201_CAM_noncl_cat_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetCAMCat(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        cluster=False,
        num_init_features=64,
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet201'])

    return model
