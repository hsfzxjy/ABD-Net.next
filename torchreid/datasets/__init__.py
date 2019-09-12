from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .market1501 import Market1501
from .market1501_d import Market1501_D
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .dukemtmcreid_d import DukeMTMCreID_D
from .msmt17 import MSMT17
from .viper import VIPeR
from .grid import GRID
from .cuhk01 import CUHK01
from .prid450s import PRID450S
from .ilids import iLIDS
from .sensereid import SenseReID
from .cub_200_2011 import CUB_200_2011

from .mars import Mars
from .veri import VeRi
from .ilidsvid import iLIDSVID
from .prid2011 import PRID2011
from .dukemtmcvidreid import DukeMTMCVidReID
from .vehicleid import VehicleID

from functools import partial

__imgreid_factory = {
    'market1501': Market1501,
    'market1501_d': Market1501_D,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'dukemtmcreid_d': DukeMTMCreID_D,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'prid450s': PRID450S,
    'ilids': iLIDS,
    'sensereid': SenseReID,
    'cub_200_2011': CUB_200_2011,
    'veri': VeRi,
    'vehicleid': VehicleID,
    **{('vehicleid_{}'.format(num)): partial(VehicleID, num) for num in (800, 1600, 2400, 3200, 6000, 13164)},
}


__vidreid_factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    'dukemtmcvidreid': DukeMTMCVidReID,
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)
