import torch.nn as nn
import torch
import math
import torch.nn.functional as F  # noqa
import torch.utils.model_zoo as model_zoo
__all__ = ['vgg16', 'vgg19', 'vgg16_bn', 'vgg19_bn']
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

class VGG(nn.Module):

    def __init__(self, num_classes, features, init_weights=True, loss={"xent"}, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.loss = loss
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def backbone_modules(self):
        return [self.features]

    def forward(self, x):
        x = self.features(x)
        v = x.view(x.size(0), -1)
        y = self.classifier(v)
        triplet_features = []
        predict_features = []
        xent_features = []
        triplet_features.append(v)
        predict_features.append(v)
        xent_features.append(y)
        return torch.cat(predict_features, 1), tuple(xent_features), tuple(triplet_features), {}
        # return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized.")
    return model

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes, **kwargs):
    model = VGG(num_classes, make_layers(cfg['D']), **kwargs)
    model = init_pretrained_weights(model, model_urls['vgg16'])
    return model


def vgg16_bn(num_classes, **kwargs):
    model = VGG(num_classes, make_layers(cfg['D'], batch_norm=True), **kwargs)
    model = init_pretrained_weights(model, model_urls['vgg16_bn'])
    return model


def vgg19(num_classes, **kwargs):
    model = VGG(num_classes, make_layers(cfg['E']), **kwargs)
    model = init_pretrained_weights(model, model_urls['vgg19'])
    return model


def vgg19_bn(num_classes, **kwargs):
    model = VGG(num_classes, make_layers(cfg['E'], batch_norm=True), **kwargs)
    model = init_pretrained_weights(model, model_urls['vgg19_bn'])
    return model
