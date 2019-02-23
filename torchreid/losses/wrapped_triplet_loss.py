import os
import torch

from .hard_mine_triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLoss


def WrappedTripletLoss(num_classes: int, use_gpu: bool, args, param_controller, htri_only=False) -> 'func':

    xent_loss = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    htri_loss = TripletLoss(margin=args.margin)

    def _loss(x, pids):

        _, y, v, features_dict = x

        if os.environ.get('sa') is not None:
            layer3, layer4_1, layer4_2 = features_dict['layers']

            layer3 = torch.norm(layer3, dim=1, p=2) ** 2 / 1024
            layer4_1 = torch.norm(layer4_1, dim=1, p=2) ** 2 / 2048
            layer4_2 = torch.norm(layer4_2, dim=1, p=2) ** 2 / 2048

            as_loss = ((layer3 - layer4_1) ** 2).sum() + ((layer3 - layer4_2) ** 2).sum()
            print(as_loss)
        else:
            as_loss = 0.

        if htri_only:
            loss = htri_loss(y, pids)
        else:
            loss = (
                args.lambda_xent * xent_loss(y, pids) +
                args.lambda_htri * htri_loss(v, pids) * param_controller.get_value()
            )

        return loss

    return _loss
