from fastai.basic_train import LearnerCallback
from fastai.callback import Callback
from torch import nn
from traitlets import List
from typing import Collection

from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


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


def create_nn_model(layer_list: list, state_size, action_size):
    """Generates an nn module.

    Notes:
        TabularModel could possibly be used along side a cnn learner instead. Will be a good idea to investigate.

    Returns:

    """
    pass