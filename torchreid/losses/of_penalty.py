from __future__ import absolute_import
from __future__ import division

import logging

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Functor:

    def __init__(self, beta):

        self.beta = beta

    def AAT_I(self, A):

        AAT = torch.bmm(A, A.permute(0, 2, 1))

        B, N, _ = AAT.size()
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa

        return AAT - I

    def spectral_norm(self, A):

        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator


class SVDO(Functor):

    def get_singular_values(self, A):

        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.spectral_norm(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.spectral_norm(AAT - I)
        return tmp + largest, largest

    def __call__(self, key, x):

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = (largest - smallest) * self.beta

        if key == 'intermediate':
            singular_penalty *= 0.01

        return singular_penalty.sum() / (x.size(0) / 32.)  # Quirk: normalize to 32-batch case


class SO(Functor):

    def __call__(self, key, x):

        batches, channels, height, width = x.size()
        x = x.view(batches, channels, -1)
        return self.beta * torch.sum(self.AAT_I(x) ** 2) ** 0.5

class DSO(Functor):

    def __call__(self, key, x):
        batches, channels, height, width = x.size()
        x = x.view(batches, channels, -1)
        xT = x.permute(0, 2, 1)
        return self.beta * (
            torch.sum(self.AAT_I(x) ** 2) ** 0.5 +
            torch.sum(self.AAT_I(xT) ** 2) ** 0.5
        )


class SRIP(Functor):

    def __call__(self, key, x):

        x = self.AAT_I(x)
        return self.beta * torch.sum(self.spectral_norm(x))


class OFPenalty(nn.Module):

    _WARNED = False

    def __init__(self, args):
        super().__init__()

        self.penalty_position = frozenset(args['of_position'])
        self.beta = args['of_beta']
        self.functor = {
            'SVDO': SVDO,
            'SO': SO,
            'SRIP': SRIP,
            'DSO': DSO
        }[args['of_type']](self.beta)

    def apply_penalty(self, k, x):

        if isinstance(x, (tuple)):
            if not len(x):
                return 0.
            return sum([self.apply_penalty(k, xx) for xx in x]) / len(x)

        return self.functor(k, x)

    def forward(self, inputs):

        _, y, _, feature_dict = inputs

        logger.debug(str(self.penalty_position))

        existed_positions = frozenset(feature_dict.keys())
        missing = self.penalty_position - existed_positions
        if missing and not self._WARNED:
            self._WARNED = True

            import warnings
            warnings.warn('OF positions {!r} are missing. IGNORED.'.format(list(missing)))

        singular_penalty = sum([self.apply_penalty(k, x) for k, x in feature_dict.items() if k in self.penalty_position])

        logger.debug(str(singular_penalty))
        return singular_penalty
