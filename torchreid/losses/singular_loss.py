from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import os

from .cross_entropy_loss import CrossEntropyLoss


USE_LOG = os.environ.get('use_log') is not None
CONSTRAINT_WEIGHTS = os.environ.get('constraint_weights') is not None
print('CONSTRAINT_WEIGHTS:', CONSTRAINT_WEIGHTS)


class SingularLoss(nn.Module):

    def __init__(self, num_classes, *, use_gpu=True, label_smooth=True, beta=None, penalty_position='before'):
        super().__init__()

        os_beta = None

        sing_beta = os.environ.get('sing_beta')
        if sing_beta is not None:
            try:
                os_beta = float(sing_beta)
            except (ValueError, TypeError):
                pass

        if os_beta is None:

            try:
                os_beta = float(os.environ.get('beta'))
            except (ValueError, TypeError):
                raise RuntimeError('No beta specified. ABORTED.')
        print('USE_GPU', use_gpu)
        self.beta = beta if not os_beta else os_beta
        print('beta', self.beta)
        self.xent_loss = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=label_smooth)
        self.penalty_position = frozenset(penalty_position.split(','))

    def dominant_eigenvalue(self, A):

        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()
        # denominator = torch.norm(
        #     x.view(B, 1, N),
        #     x
        # ).squeeze()
        # # print(denominator)

        return numerator / denominator

    def get_singular_values(self, A):

        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def apply_penalty(self, k, x):

        if isinstance(x, tuple):
            return sum([self.apply_penalty(xx) for xx in x]) / len(x)

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        if not USE_LOG:
            singular_penalty = (largest - smallest) * self.beta
        else:
            singular_penalty = (torch.log1p(largest) - torch.log1p(smallest)) * self.beta

        if k == 'layer5':
            singular_penalty *= 0.01

        return singular_penalty.sum()

    def forward(self, inputs, pids):

        _, y, _, feature_dict = inputs

        existed_positions = frozenset(feature_dict.keys())
        missing = self.penalty_position - existed_positions
        if missing:
            raise RuntimeError('Cannot apply singular loss, as positions {!r} are missing.'.format(list(missing)))

        singular_penalty = sum([self.apply_penalty(k, x) for k, x in feature_dict.items() if k in self.penalty_position])

        xloss = self.xent_loss(y, pids)
        print(singular_penalty)
        return singular_penalty + xloss
