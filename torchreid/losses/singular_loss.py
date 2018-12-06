from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

from .cross_entropy_loss import CrossEntropyLoss


USE_LOG = os.environ.get('use_log') is not None
CONSTRAINT_WEIGHTS = os.environ.get('constraint_weights') is not None


class SingularLoss(nn.Module):

    def __init__(self, num_classes, *, use_gpu=True, label_smooth=True, beta=None):
        super().__init__()

        os_beta = None

        try:
            os_beta = float(os.environ.get('beta'))
        except (ValueError, TypeError):
            raise RuntimeError('No beta specified. ABORTED.')

        self.beta = beta if not os_beta else os_beta
        self.xent_loss = CrossEntropyLoss(num_classes, use_gpu, label_smooth)

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
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N).expand(B, N, N).cuda()  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def forward(self, inputs, pids):

        x, y, _, weights = inputs

        if CONSTRAINT_WEIGHTS:
            height, width = weights.size()
            batches = 1
            W = weights.view(1, height, width)
        else:
            batches, channels, height, width = x.size()
            W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        if not USE_LOG:
            singular_penalty = (largest - smallest) * self.beta
        else:
            singular_penalty = (torch.log1p(largest) - torch.log1p(smallest)) * self.beta

        return singular_penalty.sum() + self.xent_loss(y, pids)
