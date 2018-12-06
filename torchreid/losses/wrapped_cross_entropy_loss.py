from .cross_entropy_loss import CrossEntropyLoss


class WrappedCrossEntropyLoss(CrossEntropyLoss):

    def forward(self, inputs, target):
        return super(WrappedCrossEntropyLoss, self).forward(inputs[1], target)
