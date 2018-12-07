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
            for m in module:
                yield from self.get_all_conv_layers(m)

        if isinstance(module, nn.Conv2d):
            yield module

    def forward(self, net):

        accumulator = torch.tensor(0.0).cuda()

        for conv in self.get_all_conv_layers(net.module.features):
            print(self.reg_instance(conv.weight).size())
            raise RuntimeError

        return torch.tensor(0.0).cuda()


def get_regularizer(name):

    return ConvRegularizer(mapping[name])
