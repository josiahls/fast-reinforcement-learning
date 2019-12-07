from fastai.layers import Flatten
from fastai.tabular import TabularModel
from fastai.torch_core import *


def init_cnn(mod):
    if getattr(mod, 'bias', None) is not None: nn.init.constant_(mod.bias, 0)
    if isinstance(mod, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(mod.weight)
    for sub_mod in mod.children(): init_cnn(sub_mod)


def conv_bn_lrelu(ni: int, nf: int, ks: int = 3, stride: int = 1) -> nn.Sequential:
    r""" Create a sequence Conv2d->BatchNorm2d->LeakyReLu layer. (from darknet.py) """
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks // 2),
        nn.BatchNorm2d(nf),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


class ChannelTranspose(Module):
    def forward(self, xi: Tensor):
        return xi.transpose(3, 1).transpose(3, 2)


class DQNModule(Module):

    def fix_switched_channels(self, current_channels, expected_channels, layers: list):
        if current_channels == expected_channels:
            return layers
        else:
            self.switched = True
            return [ChannelTranspose()] + layers

    def __init__(self, ni: int, ao: int, layers: Collection[int], discount: float = 0.99,
                 n_conv_blocks: Collection[int] = 0, nc=3,  emb_szs: ListSizes = None, w=-1, h=-1,
                 ks=None, stride=None):
        r"""
        Basic DQN Module.

        Args:
            ni: Number of inputs. Expecting a flat state `[1 x ni]`
            ao: Number of actions to output.
            layers: Number of layers where is determined per element.
            n_conv_blocks: If `n_conv_blocks` is not 0, then convolutional blocks will be added
                           to the head on top of existing linear layers.
            nc: Number of channels that will be expected by the convolutional blocks.
        """
        super().__init__()
        self.switched, ks = False, ifnone(ks, max((w, h)) // 100)
        stride = ifnone(stride, ks // 2)
        _layers = [conv_bn_lrelu(nc, self.nf, ks=ks, stride=stride) for self.nf in n_conv_blocks]
        self.layers = torch.nn.Sequential()

        if _layers:
            self.layers.add_module('conv_block', nn.Sequential(*(self.fix_switched_channels(ni, nc, _layers) + [Flatten()])))
            ni = int(self.layers(torch.zeros((1, w, h, nc) if self.switched else (1, nc, w, h))).view(-1,).shape[0])

        self.layers.add_module('lin_block', TabularModel(emb_szs=emb_szs, n_cont=ni, layers=layers, out_sz=ao, use_bn=True))

