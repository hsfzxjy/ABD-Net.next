from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os


class LowRankLoss(nn.Module):

    def __init__(self, beta):
        super().__init__()

        os_beta = None

        try:
            os_beta = float(os.environ.get('beta'))
        except ValueError:
            pass

        self.beta = beta if not os_beta else os_beta
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, pids):

        x, y = inputs
        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        WT = x.view(batches, channels, -1).permute(0, 2, 1)
        WWT = torch.bmm(W, WT)
        I = torch.eye(channels).expand(batches, channels, channels).cuda()  # noqa
        delta = WWT - I
        norm = torch.norm(delta.view(batches, -1), 2, 1) ** 2
        return norm.sum() * self.beta + self.xent_loss(y, pids)
