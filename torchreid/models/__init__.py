from __future__ import absolute_import

# from .resnet import *
from .resnetmid import *
from .resnext import *
from .senet import *
# from .densenet import *
from .inceptionresnetv2 import *
from .inceptionv4 import *
from .xception import *

from .nasnet import *
from .mobilenetv2 import *
from .shufflenet import *
from .squeezenet import *

from .mudeep import *
from .hacnn import *
from .pcb import *
from .vgg import *
from .mlfn import *

# from . import resnet_best
from . import resnet_mgn_abd
# from .densenet_DAN import *
# from .densenet_DAN_cat import *
# from .densenet_cl import *
# from .densenet_CAM_cat import *
# from .cltmp.densenet_cl import *

from . import densenet, resnet
from .resnet_sur import resnet50 as resnet50_nosur, resnet50_sur

__model_factory = {
    **densenet.model_mapping,
    'resnet50_sur': resnet50_sur,
    'resnet50_nosur': resnet50_nosur,
    'resnet50': resnet.resnet50,
    'resnet50_abd_old': resnet.resnet50_abd_old,
    'resnet50_abd_basel': resnet.resnet50_abd_old_baseline,
    'resnet50_abd_np2': resnet.resnet50_abd_np2,
    'resnet50_mgn_like': resnet.resnet50_mgn_like,
    # **resnet.name_function_mapping,
    'vgg19': vgg19,
    'vgg16': vgg16,
    'vgg16_128': vgg16_128,
    **resnet_mgn_abd.name_function_mapping,
    # 'resnet50_best': resnet_best.resnet50_best,
    # image classification models
    # 'resnet50': resnet50,
    # 'resnet50_fc512': resnet50_fc512,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x4d': resnext101_32x4d,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    # 'densenet121': densenet121,
    # 'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': InceptionResNetV2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2': MobileNetV2,
    'shufflenet': ShuffleNet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
