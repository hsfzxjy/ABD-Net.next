import torch
import torch.nn as nn

class ResNetOld(nn.Module):
    """
    Residual network

    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, num_classes,
                 dropout_p=None,
                 *,
                 fd_config={'parts': (), 'use_conv_head': False},
                 attention_config={'parts': ('cam', 'pam')},
                 dropout_optimizer=None,
                 fc_dims=(1024,),
                 **kwargs):

        self.sum_fusion = True
        last_stride = 1

        super().__init__()
        self.feature_dim = 2048  # 512 * block.expansion

        # backbone network
        backbone = ResNet(Bottleneck, [3, 4, 6, 3], last_stride)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.layer4_normal_branch = deepcopy(self.layer4)

        dim = fc_dims[0]
        self.dim = dim

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        num_features = 2048
        # Begin Attention Module
        self.get_attention_module()
        self.attention_config = attention_config
        # End Attention Module

        # Begin Dropout Module
        if dropout_optimizer is None:
            from torchreid.components.dropout import SimpleDropoutOptimizer
            dropout_optimizer = SimpleDropoutOptimizer(dropout_p)
        # End Dropout Module

        if os.environ.get('dropout_reduction'):
            dropout = [dropout_optimizer]
        else:
            dropout = []

        # fc_dims = [1024]
        self.fc = self._construct_fc_layer(fc_dims, num_features, dropout_optimizer)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.reduction_tr = nn.Sequential(
            nn.Conv2d(2048, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            *dropout
        )
        self._init_params(self.reduction_tr)

        try:
            part_num = int(os.environ.get('part_num'))
        except (TypeError, ValueError):
            part_num = 2

        self.part_num = part_num

        for i in range(1, part_num + 1):
            c = nn.Linear(dim, num_classes)
            setattr(self, f'classifier_p{i}', c)
            self._init_params(c)

        self._init_params(self.fc)
        self._init_params(self.classifier)

    def get_attention_module(self):

        from torchreid.components.attention import DANetHead, CAM_Module, PAM_Module

        in_channels = self.dim
        out_channels = self.dim

        self.before_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, lambda _: lambda x: x)
        self.pam_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, PAM_Module)
        self.cam_module1 = DANetHead(in_channels, out_channels, nn.BatchNorm2d, CAM_Module)
        self.sum_conv1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(out_channels, out_channels, 1))

        self._init_params(self.before_module1)
        self._init_params(self.cam_module1)
        self._init_params(self.pam_module1)
        self._init_params(self.sum_conv1)

    def backbone_modules(self):

        convs = [
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer4_normal_branch,
        ]

        return convs

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_optimizer):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """

        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []

        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(dropout_optimizer)
            # if dropout_p is not None:
            #     layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def _init_params(self, x):
        if x is None:
            return

        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, 1, 0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        layer5 = x
        x = self.layer2(x)
        x = self.layer3(x)

        triplet_features = []
        xent_features = []
        predict_features = []

        # normal branch
        x1 = x
        x1 = self.layer4_normal_branch(x1)
        x1 = self.global_avgpool(x1)
        x1 = x1.view(x1.size(0), -1)
        triplet_features.append(x1)
        x1 = self.fc(x1)
        predict_features.append(x1)
        x1 = self.classifier(x1)
        xent_features.append(x1)

        # our branch
        x2 = x
        x2 = self.layer4(x2)
        x2 = self.reduction_tr(x2)

        feature_dict = {
            'cam': [],
            'pam': [],
            'before': [],
            'after': [],
            'intermediate': layer5
        }

        margin = x2.size(2) // self.part_num

        for p in range(1, self.part_num + 1):

            f = x2[:, :, margin * (p - 1):margin * p, :]

            if not self.attention_config['parts']:
                # f_after1 = f_before1 = self.before_module1(f)
                f_after1 = f_before1 = F.relu(f)
                feature_dict['before'] = (*feature_dict['before'], f_before1)
                feature_dict['after'] = (*feature_dict['after'], f_after1)
            else:
                f_before1 = self.before_module1(f)
                f_cam1 = self.cam_module1(f)
                f_pam1 = self.pam_module1(f)

                f_sum1 = f_cam1 + f_pam1 + f_before1
                f_after1 = self.sum_conv1(f_sum1)

                feature_dict['cam'] = (*feature_dict['cam'], f_cam1)
                feature_dict['pam'] = (*feature_dict['pam'], f_pam1)
                feature_dict['before'] = (*feature_dict['before'], f_before1)
                feature_dict['after'] = (*feature_dict['after'], f_after1)

            v = self.global_avgpool(f_after1)
            v = v.view(v.size(0), -1)
            triplet_features.append(v)
            predict_features.append(v)
            v = getattr(self, f'classifier_p{p}')(v)
            xent_features.append(v)

        return torch.cat(predict_features, 1), tuple(xent_features), tuple(triplet_features), feature_dict
