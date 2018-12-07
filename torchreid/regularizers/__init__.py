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

        self.reg_instance = klass()

    def forward(self, net):

        for param in net.parameters():
            print(param)
        return torch.tensor([0.0]).cuda()


def get_regularizer(name):

    return ConvRegularizer(mapping[name])()
