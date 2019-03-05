import os
import torch
import torch.nn as nn

from .cross_entropy_loss import CrossEntropyLoss


def btr(A: 'N x C x C'):

    N, C = A.size()
    eye = torch.eye(C, device='cuda').expand(N, C, C).view(N, C * C, 1)
    return torch.bmm(A.view(N, 1, C * C), eye).view(N)


class BatchSpectralLoss(nn.Module):

    def __init__(self, num_classes, *, use_gpu=True, label_smooth=True, beta=None):
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

    def get_trace(self, A: 'N x D'):

        AAT = A @ A.permute(1, 0)

        N, _ = A.size()
        D = (AAT @ torch.ones((N, 1), device='cuda')).view(N).diag()
        return torch.trace(D - AAT)

    def apply_penalty(self, x):

        penalty = self.get_trace(x)

        return penalty.sum()

    def forward(self, inputs, pids):

        _, y, _, _ = inputs

        if not isinstance(y, tuple):
            y = (y,)

        penalty = sum([self.apply_penalty(x) for x in y])

        xloss = self.xent_loss(y, pids)
        # logger.debug(str(singular_penalty))
        print(penalty)
        return penalty + xloss