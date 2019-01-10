import torch
import torch.nn as nn

from .attention import get_attention_module_instance


class FeatureDistilationTrick(nn.Module):

    def __init__(
        self,
        # features: 'backbone of densenet121',
        parts: 'power set of "abc"',
        *,
        channels=None,
        use_conv_head: bool=False
    ):

        super().__init__()
        # self.features = features

        self.cam_modules = []
        self.channels = channels
        for part in parts:

            part: 'subset of "abc"'

            cs = []
            for key in part:
                cs.extend(channels[key])
            cs.sort()

            cam_module = get_attention_module_instance(
                'cam',
                len(cs),
                use_conv_head=use_conv_head
            )
            setattr(self, f'_cam_module_{part}', cam_module)  # force gpu

            self.cam_modules.append((cs, cam_module))

    # def forward(self, x):
    #     for index, layer in enumerate(self.features):
    #         x = layer(x)
    #         if index == 5:
    #             B, C, H, W = x.shape

    #             for cs, cam in self.cam_modules:
    #                 c_tensor = torch.tensor(cs).cuda()

    #                 new_x = x[:, c_tensor]
    #                 new_x = cam(new_x)
    #                 x[:, c_tensor] = new_x

    #     return x
