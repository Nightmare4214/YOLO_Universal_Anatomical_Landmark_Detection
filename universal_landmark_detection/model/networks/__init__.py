from .gln import GLN
from .gln2 import GLN2
from .globalNet import GlobalNet
from .loss_and_optim import *
from .tri_unet import Tri_UNet
from .u2net import U2Net
from .unet2d import UNet as unet2d
from .csnet import CsNet


def get_net(s):
    return {
        'unet2d': unet2d,
        'csnet': CsNet,
        'u2net': U2Net,
        'gln': GLN,
        'gln2': GLN2,
        'tri_unet': Tri_UNet,
        'globalnet': GlobalNet,
    }[s.lower()]


def get_loss(s):
    return {
        'l1': l1,
        'l2': l2,
        'bce': bce,
        'ac_loss': AC_loss
    }[s.lower()]


def get_optim(s):
    return {
        'adam': adam,
        'sgd': sgd,
        'adagrad': adagrad,
        'rmsprop': rmsprop,

    }[s.lower()]


def get_scheduler(s):
    return {
        'steplr': steplr,
        'multisteplr': multisteplr,
        'cosineannealinglr': cosineannealinglr,
        'reducelronplateau': reducelronplateau,
        'lambdalr': lambdalr,
        'cycliclr': cycliclr,

    }[s.lower()]
