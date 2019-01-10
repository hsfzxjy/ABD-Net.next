import torch
import torch.nn as nn

from .NR import NoneRegularizer
from .SVMO import SVMORegularizer
from .SVDO import SVDORegularizer
from .SO import SORegularizer

mapping = {
    'none': NoneRegularizer,
    'svdo': SVDORegularizer,
    'svmo': SVMORegularizer,
    'so': SORegularizer
}


class ConvRegularizer(nn.Module):

    def __init__(self, klass):
        super().__init__()
        self.reg_instance = klass()

    def get_all_conv_layers(self, module):

        if isinstance(module, (nn.Sequential, list)):
            for m in module:
                yield from self.get_all_conv_layers(m)

        if isinstance(module, nn.Conv2d):
            yield module

    def forward(self, net, ignore=False):

        accumulator = torch.tensor(0.0).cuda()

        if ignore:
            return accumulator

        for conv in self.get_all_conv_layers(net.module.backbone_convs()):
            accumulator += self.reg_instance(conv.weight)

        # print(accumulator.data)
        return accumulator


def get_regularizer(name):

    return ConvRegularizer(mapping[name])
