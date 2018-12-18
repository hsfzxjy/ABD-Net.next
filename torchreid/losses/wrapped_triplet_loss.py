from .hard_mine_triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLoss


def WrappedTripletLoss(num_classes: int, use_gpu: bool, args, htri_only=True) -> 'func':

    xent_loss = CrossEntropyLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    htri_loss = TripletLoss(margin=args.margin)

    def _loss(x, pids):

        _, y, v, _ = x

        if htri_only:
            loss = htri_loss(y, pids)
        else:
            loss = (
                args.lambda_xent * xent_loss(y, pids) +
                args.lambda_htri * htri_loss(v, pids)
            )

        return loss

    return _loss
