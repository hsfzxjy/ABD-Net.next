import os
import torch
import torch.nn as nn

from .cross_entropy_loss import CrossEntropyLoss


def btr(A: 'N x C x C'):

    result = torch.norm(A, p=1, dim=(1, 2))
    print(result)
    return result

    # N, C, _ = A.size()
    # A = torch.sqrt(torch.bmm(A.permute(0, 2, 1), A))
    # print(A)
    # eye = torch.eye(C, device='cuda').expand(N, C, C).view(N, C * C, 1)
    # return torch.bmm(A.view(N, 1, C * C), eye).view(N)


class SpectralLoss(nn.Module):

    def __init__(self, num_classes, *, use_gpu=True, label_smooth=True, beta=None, penalty_position='before'):
        super().__init__()

        os_beta = None

        sing_beta = os.environ.get('spec_beta')
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

    def get_trace(self, A: 'N x C x S'):

        N, C, _ = A.size()
        AAT = torch.bmm(A, A.permute(0, 2, 1))
        ones = torch.ones((N, C, 1), device='cuda')
        D = torch.bmm(AAT, ones).view(N, C)
        D = torch.diag_embed(D)

        return btr(D - AAT)

    def apply_penalty(self, k, x):

        if isinstance(x, tuple):
            return sum([self.apply_penalty(k, xx) for xx in x]) / len(x)

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)

        penalty = self.get_trace(W)

        if k == 'layer5':
            penalty *= 0.01

        return penalty.sum() / (x.size()[0] / 32.)  # Quirk: normalize to 32-batch case

    def forward(self, inputs, pids):

        _, y, _, feature_dict = inputs

        existed_positions = frozenset(feature_dict.keys())
        missing = self.penalty_position - existed_positions
        if missing:
            raise RuntimeError('Cannot apply singular loss, as positions {!r} are missing.'.format(list(missing)))

        penalty = sum([self.apply_penalty(k, x) for k, x in feature_dict.items() if k in self.penalty_position])

        xloss = self.xent_loss(y, pids)
        # logger.debug(str(singular_penalty))
        print(penalty)
        return penalty + xloss
