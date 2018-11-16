import torch
from torch import nn
from torch.nn import functional as F

from . import densenet as densenet_
from .attention import PAM_Module, CAM_Module

__all__ = ['densenet121_dan', 'densenet121_dan_fc512']


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
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

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat, _ = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


class DensenetDAN(densenet_.DenseNet):

    def __init__(self, num_classes, loss, fc_dims, **kwargs):

        super(DensenetDAN, self).__init__(
            num_classes=num_classes,
            loss=loss,
            num_init_features=64,
            growth_rate=32,
            block_config=(6, 12, 24, 16),
            fc_dims=None,
            dropout_p=None,
            **kwargs
        )

        self.danet_head = DANetHead(self.feature_dim, self.feature_dim, nn.BatchNorm2d)
        self.fc = self._construct_fc_layer(fc_dims, self.feature_dim * 2, dropout_p=None)
        # feature_dim changed after _construct_fc_layer, so we must construct
        # classifier again
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        f = self.features(x)
        old_f = f

        base_x = f
        # pa, pose, pose_mask = self.danet_head(base_x)
        pa = base_x
        pa = F.avg_pool2d(pa, pa.size()[2:])
        pa = pa.view(pa.size(0), -1)

        f = F.relu(f, inplace=True)
        v = self.global_avgpool(f)
        v = v.view(v.size(0), -1)

        v = torch.cat((v, pa), 1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, v
        else:
            return old_f, y, v
            # raise KeyError("Unsupported loss: {}".format(self.loss))


def densenet121_dan(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetDAN(
        num_classes=num_classes,
        loss=loss,
        fc_dims=None,
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet121'])

    return model


def densenet121_dan_fc512(num_classes, loss, pretrained='imagenet', **kwargs):

    model = DensenetDAN(
        num_classes=num_classes,
        loss=loss,
        fc_dims=[512],
        **kwargs
    )

    if pretrained == 'imagenet':
        densenet_.init_pretrained_weights(model, densenet_.model_urls['densenet121'])

    return model
