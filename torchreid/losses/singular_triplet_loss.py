from .wrapped_triplet_loss import WrappedTripletLoss
from .singular_loss import SingularLoss


def SingularTripletLoss(num_classes: int, use_gpu: bool, args) -> 'func':

    xent_loss = SingularLoss(num_classes=num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    htri_loss = WrappedTripletLoss(num_classes, use_gpu, args, htri_only=False)

    def _loss(x, pids):

        _, y, v, _ = x

        if args.htri_only:
            loss = htri_loss(x, pids)
        else:
            loss = (
                args.lambda_xent * xent_loss(x, pids) +
                args.lambda_htri * htri_loss(v, pids)
            )

        return loss

    return _loss
