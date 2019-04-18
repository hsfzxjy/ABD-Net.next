import weakref

import torch
import torch.nn as nn
from collections import defaultdict

from torchreid.utils.torchtools import init_params
from torchreid.components.attention import get_attention_module_instance

class MultiBranchNetwork(nn.Module):

    def __init__(self, backbone, args, num_classes, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.common_branch = self._get_common_branch(backbone, args)
        self.branches = self._get_branches(backbone, args)

        # for i, branch in enumerate(self.branches):
        #     self.add_module(f'branch_{i}', branch)

    def _get_common_branch(self, backbone, args):
        return NotImplemented

    def _get_branches(self, backbone, args) -> list:
        return NotImplemented

    def backbone_modules(self):

        lst = [*self.common_branch.backbone_modules()]
        for branch in self.branches:
            lst.extend(branch.backbone_modules())

        return lst

    def forward(self, x):
        x, *intermediate_fmaps = self.common_branch(x)

        fmap_dict = defaultdict(list)
        fmap_dict['intermediate'].extend(intermediate_fmaps)

        predict_features, xent_features, triplet_features = [], [], []

        for branch in self.branches:
            predict, xent, triplet, fmap = branch(x)
            predict_features.extend(predict)
            xent_features.extend(xent)
            triplet_features.extend(triplet)

            for name, fmap_list in fmap.items():
                fmap_dict[name].extend(fmap_list)

        fmap_dict = {k: tuple(v) for k, v in fmap_dict.items()}

        return torch.cat(predict_features, 1), tuple(xent_features), \
            tuple(triplet_features), fmap_dict


class Sequential(nn.Sequential):

    def backbone_modules(self):

        backbone_modules = []
        for m in self._modules:
            backbone_modules.append(m.backbone_modules())

        return backbone_modules

class GlobalBranch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['global_dim']
        self.args = args
        self.num_classes = owner.num_classes

        self._init_fc_layer()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._init_classifier()

    def backbone_modules(self):

        return []

    def _init_classifier(self):

        classifier = nn.Linear(self.output_dim, self.num_classes)
        init_params(classifier)

        self.classifier = classifier

        self.owner().classifier = classifier  # Forward Compatibility

    def _init_fc_layer(self):

        dropout_p = self.args['dropout']

        if dropout_p is not None:
            dropout_layer = [nn.Dropout(p=dropout_p)]
        else:
            dropout_layer = []

        fc = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            *dropout_layer
        )
        init_params(fc)

        self.fc = fc
        self.owner().fc = fc

    def forward(self, x):

        triplet, xent, predict = [], [], []

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        triplet.append(x)
        predict.append(x)
        x = self.classifier(x)
        xent.append(x)

        return predict, xent, triplet, {}


class ABDBranch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['abd_dim']
        self.args = args
        self.part_num = args['abd_np']
        self.num_classes = owner.num_classes

        self._init_reduction_layer()

        self.dan_module_mapping = dict()
        self._init_attention_modules()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._init_classifiers()

    def backbone_modules(self):

        return []

    def _init_classifiers(self):

        self.classifiers = []

        for p in range(1, self.part_num + 1):
            classifier = nn.Linear(self.output_dim, self.num_classes)
            init_params(classifier)
            self.classifiers.append(classifier)
            self.owner().add_module(f'classifier_p{p}', classifier)

    def _init_reduction_layer(self):

        reduction = nn.Sequential(
            nn.Conv2d(self.input_dim, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True)
        )
        init_params(reduction)

        self.reduction = reduction
        self.owner().reduction_tr = reduction

    def _init_attention_modules(self):

        args = self.args
        DAN_module_names = {'cam', 'pam'} & set(args['abd_dan'])
        use_head = not args['abd_dan_no_head']

        if not DAN_module_names:
            self.use_dan = False
            self.dan_module_mapping['before'] = lambda x: x
        else:
            self.use_dan = True
            before_module = get_attention_module_instance(
                'identity',
                self.output_dim,
                use_head=use_head
            )
            self.dan_module_mapping['before'] = before_module
            if use_head:
                init_params(before_module)
                self.owner().add_module('before_module1', before_module)

            if 'cam' in DAN_module_names:
                cam_module = get_attention_module_instance(
                    'cam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(cam_module)
                self.dan_module_mapping['cam'] = cam_module
                self.owner().add_module('cam_module1', cam_module)

            if 'pam' in DAN_module_names:
                pam_module = get_attention_module_instance(
                    'pam',
                    self.output_dim,
                    use_head=use_head
                )
                init_params(pam_module)
                self.dan_module_mapping['pam'] = pam_module
                self.owner().add_module('pam_module1', pam_module)

            sum_conv = nn.Sequential(
                nn.Dropout2d(0.1, False),
                nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1)
            )
            init_params(sum_conv)
            self.sum_conv = sum_conv
            self.owner().sum_conv1 = sum_conv

    def forward(self, x):

        predict, xent, triplet = [], [], []
        fmap = defaultdict(list)

        x = self.reduction(x)

        margin = x.size(2) // self.part_num
        for p, classifier in enumerate(self.classifiers, 1):
            x_sliced = x[:, :, margin * (p - 1):margin * p, :]

            if self.use_dan:

                to_sum = []
                for name, module in self.dan_module_mapping.items():
                    x_out = module(x_sliced)
                    to_sum.append(x_out)
                    fmap[name].append(x_out)

                fmap_after = self.sum_conv(sum(to_sum))
                fmap['after'].append(fmap_after)

            else:

                fmap_after = x_sliced
                fmap['before'].append(fmap_after)
                fmap['after'].append(fmap_after)

            v = self.avgpool(fmap_after)
            v = v.view(v.size(0), -1)
            triplet.append(v)
            predict.append(v)
            v = classifier(v)
            xent.append(v)

        return predict, xent, triplet, fmap
