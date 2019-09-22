from math import floor

import gym
import torch
from fastai.basic_train import LearnerCallback, Any
from fastai.callback import Callback
from fastai.layers import bn_drop_lin
from gym.spaces import Discrete, Box
from torch import nn
from traitlets import List
import numpy as np
from typing import Collection

from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExplorationStrategy


class BaseAgent(nn.Module):
    """
    One of the basic differences between this model type and typical openai models is that this will have its
    own learner_callbacks. This is due to the often strange and beautiful methods created for training RL agents.

    """
    def __init__(self, data: MDPDataBunch):
        super().__init__()
        self.data = data
        self.name = ''
        # Some definition of loss needs to be implemented
        self.loss = None
        self.out = None
        self.opt = None
        self.learner_callbacks = []  # type: Collection[LearnerCallback]
        # Root model that will be accessed for action decisions
        self.action_model = None  # type: nn.Module
        self.exploration_strategy = ExplorationStrategy(self.training)

    def forward(self, x):
        if isinstance(x, torch.Tensor): return x.float()
        return x

    def pick_action(self, x):
        x = self(x)
        self.out = x

        with torch.no_grad():
            if len(x.shape) > 2: raise ValueError('The agent is outputting actions with more than 1 dimension...')

            if isinstance(self.data.train_ds.env.action_space, Discrete): x = x.argmax().numpy().item()
            elif isinstance(self.data.train_ds.env.action_space, Box): x = x.squeeze(0).numpy()

            return self.exploration_strategy.perturb(x, self.data.train_ds.env.action_space)

    def interpret_q(self, items):
        raise NotImplementedError


def get_embedded(input_size, output_size, n_embeddings, n_extra_dims):
    embed = nn.Embedding(n_embeddings, n_extra_dims)
    out_size = embed(torch.clamp(torch.randn(input_size).long(), 0, 1)).shape
    return embed, np.prod(out_size)


class ToLong(nn.Module):
    def forward(self, x):
        return x.long()


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def create_nn_model(layer_list: list, action_size, state_size, use_bn=False, use_embed=True):
    """Generates an nn module.

    Notes:
        TabularModel could possibly be used along side a cnn learner instead. Will be a good idea to investigate.

    Returns:

    """
    action_size = action_size[0]  # For now the dimension of the action does not make a difference.
    # For now keep drop out as 0, test including dropout later
    ps = [0] * len(layer_list)
    sizes = [state_size] + layer_list + [action_size]
    actns = [nn.ReLU() for _ in range(len(sizes) - 2)] + [None]
    layers = []
    for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.] + ps, actns)):
        if i == 0 and use_embed:
            embedded, n_in = get_embedded(n_in[0], n_out, n_in[1], 20)
            layers += [ToLong(), embedded, Flatten()]

        layers += bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
    return nn.Sequential(*layers)

def get_next_conv_shape(c_w, c_h, stride, kernel_size):
    h = floor((c_h - kernel_size - 2) / stride) + 1 # 3 convolutional layers given (3c, 640w, 640h)
    w = floor((c_w - kernel_size - 2) / stride) + 1
    return h, w


def get_conv(input_tuple, act, kernel_size, stride, n_conv_layers, layers):
    """
    Useful guideline for convolutional net shape change:

    Shape:
    - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
    - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

      .. math::
          H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

      .. math::
          W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor


    :param input_tuple:
    :param act:
    :param kernel_size:
    :param stride:
    :param n_conv_layers:
    :param layers:
    :return:
    """
    h, w = input_tuple[0], input_tuple[1]
    conv_layers = []
    for i in range(n_conv_layers):
        h, w = get_next_conv_shape(h, w, stride, kernel_size)
        conv_layers.append(torch.nn.Conv2d(input_tuple[2], 3, kernel_size=kernel_size, stride=stride))
        conv_layers.append(act)
    return layers + conv_layers, 3 * (h + 1) * (w + 1)



def create_cnn_model(layer_list: list, action_size, state_size, use_bn=False, kernel_size=5, stride=3, n_conv_layers=3):
    """Generates an nn module.

    Notes:
        TabularModel could possibly be used along side a cnn learner instead. Will be a good idea to investigate.

    Returns:

    """
    # For now keep drop out as 0, test including dropout later
    ps = [0] * len(layer_list)
    sizes = [state_size] + layer_list + [action_size]
    actns = [nn.ReLU() for _ in range(n_conv_layers + len(sizes) - 2)] + [None]
    layers = []
    for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.] + ps, actns)):
        if type(n_in) == tuple:
            layers, n_in = get_conv(n_in, act, kernel_size, n_conv_layers=n_conv_layers, layers=layers, stride=stride)
            layers += [Flatten()]

        layers += bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
    return nn.Sequential(*layers)