import torch
import torch.nn as nn

from .NR import NoneRegularizer
from .SVO import SVORegularizer

mapping = {
    'none': NoneRegularizer,
    'svo': SVORegularizer
}


class ConvRegularizer(nn.Module):

    def __init__(self, klass):
        super().__init__()
        self.reg_instance = klass()

    def get_all_conv_layers(self, module):

        if isinstance(module, nn.Sequential):
            yield from self.get_all_conv_layers(module)

        if isinstance(module, nn.Conv2d):
            yield module

    def forward(self, net):
        print(net._modules)
        print(list(self.get_all_conv_layers(net.features)))

        return torch.tensor([0.0]).cuda()


def get_regularizer(name):

    return ConvRegularizer(mapping[name])
