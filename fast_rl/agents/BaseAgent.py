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
    own callbacks. This is due to the often strange and beautiful methods created for training RL agents.

    """
    def __init__(self, data: MDPDataBunch):
        super().__init__()
        self.data = data
        self.callbacks = []  # type: Collection[LearnerCallback]
        # Root model that will be accessed for action decisions
        self.action_model = None  # type: nn.Module
        self.exploration_strategy = ExplorationStrategy(self.training)

    def forward(self, x):
        if isinstance(x, torch.Tensor): return x.float()
        return x

    def pick_action(self, x):
        x = self(x)

        with torch.no_grad():
            if isinstance(self.data.train_ds.env.action_space, Discrete): x = x.argmax().numpy().item()
            elif isinstance(self.data.train_ds.env.action_space, Box): x = x.squeeze(0).numpy()

            if len(x.shape) > 1: raise ValueError('The agent is outputting actions with more than 1 dimension...')
            return self.exploration_strategy.perturb(x, self.data.train_ds.env.action_space)


def create_nn_model(layer_list: list, action_size, state_size, use_bn=False):
    """Generates an nn module.

    Notes:
        TabularModel could possibly be used along side a cnn learner instead. Will be a good idea to investigate.

    Returns:

    """
    # For now keep drop out as 0, test including dropout later
    ps = [0] * len(layer_list)
    sizes = [state_size] + layer_list + [action_size]
    actns = [nn.Tanh() for _ in range(len(sizes) - 2)] + [None]
    layers = []
    for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.] + ps, actns)):
        layers += bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
    return nn.Sequential(*layers)
