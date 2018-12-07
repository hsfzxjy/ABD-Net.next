from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

w_rate = 1e-4


class SVORegularizer(nn.Module):

    def __init__(self):
        super().__init__()

        os_beta = None

        try:
            os_beta = float(os.environ.get('beta'))
        except (ValueError, TypeError):
            raise RuntimeError('No beta specified. ABORTED.')
        self.beta = os_beta

    def dominant_eigenvalue(self, A: 'N x N'):

        N, _ = A.size()
        x = torch.rand(N, 1).cuda()

        Ax = A @ x
        AAx = A @ Ax

        return AAx.T @ Ax / (Ax.T @ Ax)

        # for _ in range(1):
        #     x = A @ x
        # numerator = torch.bmm(
        #     torch.bmm(A, x).permute(0, 2, 1),
        #     x
        # ).squeeze()
        # denominator = torch.bmm(
        #     x.permute(0, 2, 1),
        #     x
        # ).squeeze()

        # return numerator / denominator

    def get_singular_values(self, A: 'M x N, M >= N'):

        ATA = A.transpose() @ A
        N, _ = ATA.size()
        largest = self.dominant_eigenvalue(ATA)
        I = torch.eye(N).cuda()  # noqa
        I = I * largest  # noqa
        tmp = self.dominant_eigenvalue(ATA - I)
        return tmp + largest, largest

    def forward(self, W: 'S x C x H x W'):

        old_W = W
        old_size = W.size()

        W = W.view(old_size[0], -1).transpose()
        # W = W.permute(2, 3, 0, 1).view(old_size[0] * old_size[2] * old_size[3], old_size[1])

        smallest, largest = self.get_singular_values(W)
        return self.beta * (largest - smallest) + w_rate * old_W.sum()
