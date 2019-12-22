from math import floor

from fastai.basic_train import LearnerCallback
from fastai.torch_core import *
from gym.spaces import Discrete, Box

from fast_rl.core.agent_core import ExplorationStrategy
from fast_rl.core.data_block import MDPDataBunch


# class BaseAgent(nn.Module):
#     r"""
#     One of the basic differences between this model type and typical Openai models is that this will have its
#     own learner_callbacks. This is due to the often strange and beautiful methods created for training RL agents.
#     """
#     def __init__(self, data: MDPDataBunch):
#         super().__init__()
#         self.data = data
#         self.name = ''
#         # Some definition of loss needs to be implemented
#         self.loss = None
#         self.out = None
#         self.opt = None
#         self.warming_up = False
#         self.learner_callbacks = []  # type: Collection[LearnerCallback]
#         # Root model that will be accessed for action decisions
#         self.action_model = None  # type: nn.Module
#         self.exploration_strategy = ExplorationStrategy(self.training)
#
#     def forward(self, x):
#         if isinstance(x, torch.Tensor): return x.float()
#         return x
#
#     def pick_action(self, x):
#         x = self(x)
#         self.out = x
#
#         with torch.no_grad():
#             if len(x.shape) > 2: raise ValueError('The agent is outputting actions with more than 1 dimension...')
#
#             if isinstance(self.data.train_ds.env.action_space, Discrete): action = x.argmax().cpu().numpy().item()
#             elif isinstance(self.data.train_ds.env.action_space, Box) and len(x.shape) != 1: action = x.squeeze(0).cpu().numpy()
#
#             action = self.exploration_strategy.perturb(action, self.data.train_ds.env.action_space)
#
#             return action
#
#     def interpret_q(self, items):
#         raise NotImplementedError


def get_embedded(input_size, output_size, n_embeddings, n_extra_dims):
    embed = nn.Embedding(n_embeddings, n_extra_dims)
    out_size = embed(torch.clamp(torch.randn(input_size).long(), 0, 1)).shape
    return embed, np.prod(out_size)


class ToLong(nn.Module):
    def forward(self, x):
        return x.long()


class SwapImageChannel(nn.Module):
    def forward(self, x):
        return x.transpose(1, 3)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def get_next_conv_shape(c_w, c_h, stride, kernel_size):
    h = floor((c_h - kernel_size - 2) / stride) + 1 # 3 convolutional layers given (3c, 640w, 640h)
    w = floor((c_w - kernel_size - 2) / stride) + 1
    return h, w


def get_conv(input_tuple, act, kernel_size, stride, n_conv_layers, layers):
    r"""
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


    Args:
        input_tuple:
        act:
        kernel_size:
        stride:
        n_conv_layers:
        layers:
    """
    h, w = input_tuple[1], input_tuple[2]
    conv_layers = [SwapImageChannel()]
    for i in range(n_conv_layers):
        h, w = get_next_conv_shape(h, w, stride, kernel_size)
        conv_layers.append(torch.nn.Conv2d(input_tuple[3], 3, kernel_size=kernel_size, stride=stride))
        conv_layers.append(act)

    output_size = torch.prod(torch.tensor(nn.Sequential(*(layers + conv_layers))(torch.rand(input_tuple)).shape))
    return layers + conv_layers, output_size
