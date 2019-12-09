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


class TabularEmbedWrapper(Module):
    def __init__(self, tabular_model: Module):
        super().__init__()
        self.tabular_model = tabular_model

    def forward(self, xi: Tensor, *args):
        return self.tabular_model(xi, xi)


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
        self.layers = nn.Sequential()
        _layers = [conv_bn_lrelu(nc, self.nf, ks=ks, stride=stride) for self.nf in n_conv_blocks]

        if _layers: ni = self.setup_conv_block(_layers=_layers, ni=ni, nc=nc, w=w, h=h)
        self.setup_linear_block(_layers=_layers, ni=ni, nc=nc, w=w, h=h, emb_szs=emb_szs, layers=layers, ao=ao)

    def setup_conv_block(self, _layers, ni, nc, w, h):
        self.layers.add_module('conv_block', nn.Sequential(*(self.fix_switched_channels(ni, nc, _layers) + [Flatten()])))
        return int(self.layers(torch.zeros((1, w, h, nc) if self.switched else (1, nc, w, h))).view(-1, ).shape[0])

    def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao):
        tabular_model = TabularModel(emb_szs=emb_szs, n_cont=ni, layers=layers, out_sz=ao, use_bn=False)
        if not emb_szs: tabular_model.embeds = None
        self.layers.add_module('lin_block', TabularEmbedWrapper(tabular_model))

    def forward(self, xi: Tensor):
        return self.layers(xi)


class FixedTargetDQNModule(Module):
    def __init__(self, ni: int, ao: int, layers: Collection[int], tau=1, sub_module_cls=DQNModule, **kwargs):
        super().__init__()
        self.tau = tau

        self.action_model: Module = sub_module_cls(ni=ni, ao=ao, layers=layers, **kwargs)
        self.target_model: Module = copy(self.action_model)

    def forward(self, xi: Tensor):
        return self.action_model(xi)

    def target_copy_over(self):
        r""" Updates the target network from calls in the FixedTargetDQNTrainer callback."""
        # self.target_net.load_state_dict(self.action_model.state_dict())
        for target_param, local_param in zip(self.target_model.parameters(), self.action_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def calc_y(self, s_prime, masking, r, y_hat):
        r"""
        Uses the equation:

        .. math::

                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]

        """
        return self.discount * self.target_modela(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)


class DoubleDQNModule(FixedTargetDQNModule):
    def calc_y(self, s_prime, masking, r, y_hat):
        return self.discount * self.target_model(s_prime).gather(1, self.action_model(s_prime).argmax(1).unsqueeze(
            1)) * masking + r.expand_as(y_hat)


class DuelingBlock(nn.Module):
    def __init__(self, ao, stream_input_size):
        super().__init__()

        self.val = nn.Linear(stream_input_size, 1)
        self.adv = nn.Linear(stream_input_size, ao)

    def forward(self, xi):
        r"""Splits the base neural net output into 2 streams to evaluate the advantage and v of the s space and
        corresponding actions.

        .. math::
           Q(s,a;\; \Theta, \\alpha, \\beta) = V(s;\; \Theta, \\beta) + A(s, a;\; \Theta, \\alpha) - \\frac{1}{|A|}
           \\Big\\sum_{a'} A(s, a';\; \Theta, \\alpha)

        """
        val, adv = self.val(xi), self.adv(xi)
        xi = val.expand_as(adv) + (adv - adv.mean()).squeeze(0)
        return xi


class DuelingDQNModule(DQNModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao):
        model = TabularModel(emb_szs=emb_szs, n_cont=ni, layers=layers, out_sz=ao, use_bn=False)
        if not emb_szs: model.embeds = None
        model.layers, removed_layer = split_model(model.layers, [last_layer(model)])
        ni = removed_layer[0].in_features
        self.layers.add_module('lin_block', TabularEmbedWrapper(model))
        self.layers.add_module('dueling_block', DuelingBlock(ao, ni))


class DuelingDQNFixedTargetModule(FixedTargetDQNModule):
    def __init__(self, **kwargs):
        super().__init__(sub_module_cls=DuelingDQNModule, **kwargs)


class DoubleDuelingModule(DoubleDQNModule):
    def __init__(self, **kwargs):
        super().__init__(sub_module_cls=DuelingDQNModule, **kwargs)
