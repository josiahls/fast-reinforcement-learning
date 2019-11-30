from copy import deepcopy
from functools import partial
from typing import Tuple

import numpy as np
import torch
from fastai.basic_train import LearnerCallback, Any, F, OptimWrapper, ifnone
from torch import optim, nn

from fast_rl.agents.agents_base import BaseAgent, ToLong, get_embedded, Flatten, get_conv
from fast_rl.core.data_block import MDPDataBunch, MDPDataset, State, Action
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon


class BaseDQNCallback(LearnerCallback):
    def __init__(self, learn, max_episodes=None):
        r"""Handles basic DQN end of step model optimization."""
        super().__init__(learn)
        self.n_skipped = 0
        self._persist = max_episodes is not None
        self.max_episodes = max_episodes
        self.episode = -1
        self.iteration = 0
        # For the callback handler
        self._order = 0
        self.previous_item = None

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs if not self._persist else self.max_episodes

    def on_epoch_begin(self, epoch, **kwargs: Any):
        self.episode = epoch if not self._persist else self.episode + 1
        self.iteration = 0

    def on_loss_begin(self, **kwargs: Any):
        r"""Performs memory updates, exploration updates, and model optimization."""
        if self.learn.model.training: self.learn.model.memory.update(item=self.learn.data.train_ds.x.items[-1])
        self.learn.model.exploration_strategy.update(self.episode, max_episodes=self.max_episodes,
                                                     do_exploration=self.learn.model.training)
        post_optimize = self.learn.model.optimize()
        if self.learn.model.training: self.learn.model.memory.refresh(post_optimize=post_optimize)
        self.iteration += 1


class FixedTargetDQNCallback(LearnerCallback):
    def __init__(self, learn, copy_over_frequency=3):
        r"""Handles updating the target model in a fixed target DQN.

        Args:
            learn: Basic Learner.
            copy_over_frequency: For every N iterations we want to update the target model.
        """
        super().__init__(learn)
        self._order = 1
        self.iteration = 0
        self.copy_over_frequency = copy_over_frequency

    def on_step_end(self, **kwargs: Any):
        self.iteration += 1
        if self.iteration % self.copy_over_frequency == 0 and self.learn.model.training:
            self.learn.model.target_copy_over()


class StateNorm(nn.Module):
    def __init__(self, state: State, device='cpu'):
        super().__init__()
        self.minimum = torch.from_numpy(np.array(state.bounds.min)).float().to(device=device)
        self.maximum = torch.from_numpy(np.array(state.bounds.max)).float().to(device=device)

    def forward(self, x):
        return (x - self.minimum.expand_as(x)) / (self.maximum.expand_as(x) - self.minimum.expand_as(x))


def get_action_dqn_fully_conn(layers, action: Action, state: State, activation=nn.ReLU, embed=False, normalize=False,
                              device='cpu'):
    module_layers = []
    if normalize: module_layers.append(StateNorm(state, device))
    for i, size in enumerate(layers):
        if i == 0:
            if embed:
                embedded, out = get_embedded(state.s.shape[1], size, state.n_possible_values, 5)
                module_layers += [ToLong(), embedded, Flatten(), nn.Linear(out, size)]
            else:
                module_layers.append(nn.Linear(state.s.shape[1], size))
        else:
            module_layers.append(nn.Linear(layers[i - 1], size))
        module_layers.append(activation())

    module_layers.append(nn.Linear(layers[-1], action.n_possible_values))
    return nn.Sequential(*module_layers).to(device=device)


def get_action_dqn_cnn(layers, action: Action, state: State, activation=nn.ReLU, kernel_size=5, stride=2):
    module_layers, out_size = get_conv(state.s.shape, activation(), kernel_size=kernel_size, stride=stride,
                                       n_conv_layers=3, layers=[])
    module_layers += [Flatten()]
    layers.append(action.n_possible_values)
    for i, layer in enumerate(layers):
        module_layers += [nn.Linear(out_size, layer)] if i == 0 else [nn.Linear(layers[i - 1], layer)]
        if i != len(layers) - 1: module_layers += [activation()]

    return nn.Sequential(*module_layers)


class DQN(BaseAgent):
    def __init__(self, data: MDPDataBunch, memory=None, lr=0.00025, discount=0.95, grad_clip=5,
                 max_episodes=None, exploration_strategy=None, use_embeddings=False, layers=None,
                 loss_func=None, opt=None, attempt_normalize=True):
        """Trains an Agent using the Q Learning method on a neural net.

        Notes:
            This is not a true implementation of [1]. A true implementation uses a fixed target network.

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).

        Args:
            data: Used for size input / output information.
        """
        super().__init__(data)
        # TODO add recommend cnn based on s size?
        self.attempt_normalize = attempt_normalize
        self.name = 'DQN'
        self.use_embeddings = use_embeddings
        self.batch_size = data.train_ds.bs
        self.discount = discount
        self.warming_up = True
        self.lr = lr
        self.gradient_clipping_norm = grad_clip
        self.loss_func = ifnone(loss_func, F.mse_loss)
        self.memory = ifnone(memory, ExperienceReplay(10000))
        self.action_model = self.initialize_action_model(ifnone(layers, [64, 64]), data.train_ds)
        self.opt = OptimWrapper.create(ifnone(optim.Adam, opt), lr=self.lr, layer_groups=[self.action_model])
        self.learner_callbacks += [partial(BaseDQNCallback, max_episodes=max_episodes)] + self.memory.callbacks
        self.exploration_strategy = ifnone(exploration_strategy, GreedyEpsilon(epsilon_start=1, epsilon_end=0.1,
                                                                               decay=0.001,
                                                                               do_exploration=self.training))

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def initialize_action_model(self, layers, data):
        if len(data.state.s.shape) == 4 and data.state.s.shape[-1] < 4:
            model = get_action_dqn_cnn(deepcopy(layers), data.action, data.state, kernel_size=5, stride=2)
        else: model = get_action_dqn_fully_conn(deepcopy(layers), data.action, data.state, embed=self.use_embeddings,
                                                normalize=self.attempt_normalize, device=self.data.device)
        model.apply(self.init_weights)
        return model

    def forward(self, x):
        x = super(DQN, self).forward(x)
        return self.action_model(x)

    def sample_mask(self) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        self.warming_up = False
        # Perhaps have memory as another itemlist? Should investigate.
        with torch.no_grad():
            sampled = self.memory.sample(self.batch_size)

            r = torch.cat([item.reward.float() for item in sampled]).to(self.data.device)
            s_prime = torch.cat([item.s_prime.float() for item in sampled]).to(self.data.device)
            s = torch.cat([item.s.float() for item in sampled]).to(self.data.device)
            a = torch.cat([item.a.long() for item in sampled]).to(self.data.device)
            d = torch.cat([item.done.float() for item in sampled]).to(self.data.device)

        masking = torch.sub(1.0, d).to(self.data.device)
        return r, s_prime, s, a, d, masking

    def calc_y_hat(self, s, a): return self.action_model(s).gather(1, a)

    def calc_y(self, s_prime, masking, r, y_hat):
        return self.discount * self.action_model(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)

    def optimize(self):
        r"""Uses ER to optimize the Q-net (without fixed targets).
        
        Uses the equation:

        .. math::
                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]

        
        Returns (dict): Optimization information

        """
        if len(self.memory) > self.batch_size:
            r, s_prime, s, a, d, masking = self.sample_mask()

            # Traditional `maze-random-5x5-v0` with have a model output a Nx4 output.
            # since r is just Nx1, we spread the reward into the actions.
            y_hat = self.calc_y_hat(s, a)
            y = self.calc_y(s_prime, masking, r, y_hat)

            loss = self.loss_func(y, y_hat)

            if self.training:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), self.gradient_clipping_norm)
                for param in self.action_model.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.opt.step()

            with torch.no_grad():
                self.loss = loss
                post_info = {'td_error': (y - y_hat).cpu().numpy()}
                return post_info

    def interpret_q(self, items):
        with torch.no_grad():
            s = torch.cat([item.s.float() for item in items])
            a = torch.cat([item.a.long() for item in items])
            return self.action_model(s).gather(1, a)


class FixedTargetDQN(DQN):
    def __init__(self, data: MDPDataBunch, memory=None, tau=0.01, copy_over_frequency=3, **kwargs):
        r"""Trains an Agent using the Q Learning method on a 2 neural nets.

        Notes:
            Unlike the base DQN, this is a true reflection of ref [1]. We use 2 models instead of one to allow for
            training the action model more stably.

        Args:
            data: Used for size input / output information.

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).
        """
        super().__init__(data, memory, **kwargs)
        self.name = 'DQN Fixed Targeting'
        self.tau = tau
        self.target_net = deepcopy(self.action_model)
        self.learner_callbacks += [partial(FixedTargetDQNCallback, copy_over_frequency=copy_over_frequency)]

    def target_copy_over(self):
        r""" Updates the target network from calls in the FixedTargetDQNCallback callback."""
        # self.target_net.load_state_dict(self.action_model.state_dict())
        for target_param, local_param in zip(self.target_net.parameters(), self.action_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def calc_y(self, s_prime, masking, r, y_hat):
        r"""
        Uses the equation:

        .. math::

                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]

        """
        return self.discount * self.target_net(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)


class DoubleDQN(FixedTargetDQN):
    def __init__(self, data: MDPDataBunch, memory=None, copy_over_frequency=3, **kwargs):
        r"""
        Double DQN training.

        References:
            [1] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning."
            Thirtieth AAAI conference on artificial intelligence. 2016.

        Args:
            data: Used for size input / output information.
        """
        super().__init__(data=data, memory=memory, copy_over_frequency=copy_over_frequency, **kwargs)
        self.name = 'DDQN'

    def calc_y(self, s_prime, masking, r, y_hat):
        return self.discount * self.target_net(s_prime).gather(1, self.action_model(s_prime).argmax(1).unsqueeze(
            1)) * masking + r.expand_as(y_hat)


class DuelingDQNModule(nn.Module):
    def __init__(self, action, stream_input_size):
        super().__init__()

        self.val = nn.Linear(stream_input_size, 1)
        self.adv = nn.Linear(stream_input_size, action.n_possible_values)

    def forward(self, x):
        r"""Splits the base neural net output into 2 streams to evaluate the advantage and v of the s space and
        corresponding actions.

        .. math::
           Q(s,a;\; \Theta, \\alpha, \\beta) = V(s;\; \Theta, \\beta) + A(s, a;\; \Theta, \\alpha) - \\frac{1}{|A|}
           \\Big\\sum_{a'} A(s, a';\; \Theta, \\alpha)

        """
        val, adv = self.val(x), self.adv(x)
        x = val.expand_as(adv) + (adv - adv.mean()).squeeze(0)
        return x


class DuelingDQN(FixedTargetDQN):
    def __init__(self, data: MDPDataBunch, memory=None, **kwargs):
        r"""Replaces the basic action model with a DuelingDQNModule which splits the basic model into 2 streams.

        References:
            [1] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning."
            arXiv preprint arXiv:1511.06581 (2015).

        Args:
            data:
        """
        super().__init__(data, memory, **kwargs)
        self.name = 'Dueling DQN'

    def initialize_action_model(self, layers, data):
        base = super().initialize_action_model(layers, data)[:-2]
        dueling_head = DuelingDQNModule(action=data.action, stream_input_size=base[-1].out_features)
        return nn.Sequential(base, dueling_head)


class DoubleDuelingDQN(DoubleDQN, DuelingDQN):
    def __init__(self, data: MDPDataBunch, memory=None, **kwargs):
        r"""
        Combines both Dueling DQN and DDQN.

        Args:
            data: Used for size input / output information.
        """
        super().__init__(data, memory, **kwargs)
        self.name = 'DDDQN'
