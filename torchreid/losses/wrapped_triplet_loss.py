import os
import torch

from .hard_mine_triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLoss


def WrappedTripletLoss(num_classes: int, use_gpu: bool, args, param_controller, htri_only=False) -> 'func':

    xent_loss = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    htri_loss = TripletLoss(margin=args.margin)

    def _loss(x, pids):

        _, y, v, features_dict = x

        from .as_loss import as_loss

        as_loss_value = as_loss(features_dict)

        if htri_only:
            loss = htri_loss(y, pids)
        else:
            loss = (
                args.lambda_xent * xent_loss(y, pids) +
                args.lambda_htri * htri_loss(v, pids) * param_controller.get_value()
            ) + as_loss_value

        return loss

    return _loss
