import torch
import torch.nn as nn


class NoneRegularizer(nn.Module):

    def forward(self, _):
        return torch.tensor(0.0).cuda()
