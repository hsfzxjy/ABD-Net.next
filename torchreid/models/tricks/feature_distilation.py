import torch
import torch.nn as nn

channels = {
    'a': [
        7, 9, 20, 24, 28, 36, 38, 42, 44, 52, 61, 63, 66, 74, 77, 80, 89, 105, 108, 109, 119, 120
    ],

    'b': [2, 6, 14, 17, 23, 29, 30, 33, 34, 35, 39, 43, 48, 51, 62, 72, 81, 92, 101, 103, 115, 123, 127],
    'c': [0, 1, 3, 4, 5, 8, 10, 11, 12, 13, 15, 16, 18, 19, 21, 22, 25, 26, 27, 31, 32, 37, 40, 41, 45, 46, 47, 49, 50, 53, 54, 55, 56, 57, 58, 59, 60, 64, 65, 67, 68, 69, 70, 71, 73, 75, 76, 78, 79, 82, 83, 84, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 102, 104, 106, 107, 110, 111, 112, 113, 114, 116, 117, 118, 121, 122, 124, 125, 126]
}

from .attention import get_attention_module_instance


class FeatureDistilationTrick(nn.Module):

    def __init__(
        self,
        features: 'backbone of densenet121',
        parts: 'power set of "abc"',
        *,
        use_conv_head: bool=False
    ):

        super().__init__()
        self.features = features

        self.cam_modules = []
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

    def forward(self, x):

        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 5:
                B, C, H, W = x.shape

                for cs, cam in self.cam_modules:
                    c_tensor = torch.tensor(cs).cuda()

                    new_x = x[:, c_tensor]
                    new_x = cam(new_x)
                    x[:, c_tensor] = new_x

        return x
