import torch
from .wrapped_triplet_loss import WrappedTripletLoss
from .singular_loss import SingularLoss

import os


def SingularTripletLoss(num_classes: int, use_gpu: bool, args, param_controller) -> 'func':

    xent_loss = SingularLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth, penalty_position=args.penalty_position)
    htri_loss = WrappedTripletLoss(num_classes, use_gpu, args, param_controller, htri_only=True)

    def _loss(x, pids):

        _, y, v, features_dict = x

        if os.environ.get('sa') is not None:
            layer3, layer4_1, layer4_2 = features_dict['layers']

            layer3 = torch.norm(layer3, dim=1, p=2) ** 2 / 1024
            layer4_1 = torch.norm(layer4_1, dim=1, p=2) ** 2 / 2048
            layer4_2 = torch.norm(layer4_2, dim=1, p=2) ** 2 / 2048

            as_loss = ((layer3 - layer4_1) ** 2).sum() + ((layer3 - layer4_2) ** 2).sum()
        else:
            as_loss = 0.

        loss = (
            args.lambda_xent * xent_loss(x, pids) +
            args.lambda_htri * htri_loss(x, pids) * param_controller.get_value()
        )

        return loss + as_loss

    return _loss
