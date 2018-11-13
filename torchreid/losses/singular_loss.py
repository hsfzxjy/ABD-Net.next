from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

USE_LOG = os.environ.get('use_log') is not None


class SingularLoss(nn.Module):

    def __init__(self, beta):
        super().__init__()

        os_beta = None

        try:
            os_beta = float(os.environ.get('beta'))
        except (ValueError, TypeError):
            pass

        self.beta = beta if not os_beta else os_beta
        self.xent_loss = nn.CrossEntropyLoss()

    def dominant_eigenvalue(self, A):

        B, N, _ = A.size()
        x = torch.ones(B, N, 1).cuda()

        for _ in range(1):
            x = torch.bmm(A, x)
        numerator = torch.bmm(
            torch.bmm(A, x).permute(0, 2, 1),
            x
        ).squeeze()
        denominator = torch.bmm(
            x.permute(0, 2, 1),
            x
        ).squeeze()
        # print(denominator)

        return numerator / denominator

    def get_singular_values(self, A):

        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        # print('largest')
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N).expand(B, N, N).cuda()  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        # print('small')
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def forward(self, inputs, pids):

        x, y = inputs

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        # ones = torch.ones(batches).cuda()
        # singular_penalty = (smallest / largest - ones) ** 2 * self.beta
        if not USE_LOG:
            singular_penalty = (largest - smallest) * self.beta
        else:
            singular_penalty = (torch.log1p(largest) - torch.log1p(smallest)) * self.beta
        # singular_penalty = (torch.ones(batches).cuda() - torch.exp(-singular_penalty / self.beta))

        # WT = x.view(batches, channels, -1).permute(0, 2, 1)
        # WWT = torch.bmm(W, WT)
        # I = torch.eye(channels).expand(batches, channels, channels).cuda()  # noqa
        # delta = WWT - I
        # low_rank_penalty = torch.norm(delta.view(batches, -1), 2, 1) ** 2

        # print('penalty', singular_penalty.data.tolist())
        return singular_penalty.sum() + self.xent_loss(y, pids)
