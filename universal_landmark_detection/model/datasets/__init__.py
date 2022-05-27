from .cephalometric import Cephalometric
from .chest import Chest
from .hand import Hand


def get_dataset(s):
    return {
        'cephalometric': Cephalometric,
        'hand': Hand,
        'chest': Chest,
    }[s.lower()]
