import torch
import torch.nn as nn
import torch.nn.functional as F


class IncidenceLoss(nn.Module):

    def forward(self, x, pids):

        _, _, features, _ = x
        features: 'N x C x W x H'

        W = features.view(features.size(0), -1)
        W = F.normalize(W, p=2, dim=1)
        WWT = W @ W.permute(1, 0)

        targets = pids.data.cpu()
        A = [
            [1. if i == j else 0. for j in targets]
            for i in targets
        ]
        A = torch.tensor(A).cuda()
        print('WWT', WWT.size())
        print('A', A.size())

        return ((WWT - A)**2).sum() ** (1 / 2)
