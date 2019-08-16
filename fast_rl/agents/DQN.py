from copy import deepcopy

import torch
from fastai.basic_train import LearnerCallback, Any, F
from torch import optim
from torch.nn import MSELoss
import numpy as np

from fast_rl.agents.BaseAgent import BaseAgent, create_nn_model
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon


class BaseDQNCallback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.max_episodes = 0
        self.episode = 0
        self.iteration = 0

    def on_train_begin(self, max_episodes, **kwargs: Any):
        self.max_episodes = max_episodes

    def on_epoch_begin(self, episode, **kwargs: Any):
        self.episode = episode
        self.iteration = 0

    def on_step_end(self, learn: AgentLearner, **kwargs: Any):
        learn.model.memory.update(learn.data.x.items[-1])
        learn.model.exploration_strategy.update(self.episode, self.max_episodes, do_exploration=learn.model.training)
        learn.model.optimize()
        self.iteration += 1


class FixedTargetCallback(BaseDQNCallback):
    def __init__(self, learn, copy_over_frequency=3):
        super().__init__(learn)
        self.copy_over_frequency = copy_over_frequency

    def on_epoch_end(self, learn: AgentLearner, **kwargs: Any):
        if self.episode % self.copy_over_frequency == 0:
            learn.model.target_copy_over()


class DQN(BaseAgent):
    def __init__(self, data: MDPDataBunch):
        """Trains an Agent using the Q Learning method on a neural net.

        Notes:
            This is not a true implementation of [1]. A true implementation uses a fixed target network.

        Args:
            data:

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).
        """
        super().__init__(data)
        # TODO add recommend cnn based on state size?
        self.batch_size = 64
        self.discount = 0.99
        self.lr = 0.001
        self.loss_func = F.smooth_l1_loss
        self.memory = ExperienceReplay(10000)
        self.action_model = create_nn_model([10, 10, 10], *data.get_action_state_size())
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=self.lr)
        self.callbacks += [BaseDQNCallback(self)]
        self.exploration_strategy = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.01,
                                                  do_exploration=self.training)

    def forward(self, x):
        x = super(DQN, self).forward(x)
        return self.action_model(x)

    def optimize(self):
        """
        Uses ER to optimize the Q-net (without fixed targets).
        
        Uses the equation:

        .. math::
                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]

        
        Returns:
        """
        if len(self.memory) == self.memory.max_size:
            # Perhaps have memory as another itemlist? Should investigate.
            sampled = self.memory.sample(self.batch_size)
            with torch.no_grad():
                r = torch.from_numpy(np.array([item.reward for item in sampled])).float()
                s_prime = torch.from_numpy(np.array([item.result_state for item in sampled])).float()
                s = torch.from_numpy(np.array([item.current_state for item in sampled])).float()
                a = torch.from_numpy(np.array([item.actions for item in sampled])).long()

            # Traditional `maze-random-5x5-v0` with have a model output a Nx4 output.
            # since r is just Nx1, we spread the reward into the actions.
            y_hat = self.action_model(s).gather(1, a)
            y = self.discount * self.action_model(s_prime).max(axis=1)[0] + r.expand_as(y_hat)

            loss = self.loss_func(y, y_hat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


class FixedTargetDQN(DQN):
    def __init__(self, data: MDPDataBunch):
        super().__init__(data)
        self.target_net = deepcopy(self.action_model)
        self.callbacks += [FixedTargetCallback(self, 10)]

    def target_copy_over(self):
        self.target_net.load_state_dict(self.action_model.state_dict())

    def optimize(self):
        """
        Uses ER to optimize the Q-net.

        Uses the equation:

        .. math::
                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
                \;|\; s, a \Big]


        Returns:
        """
        if len(self.memory) == self.memory.max_size:
            # Perhaps have memory as another itemlist? Should investigate.
            sampled = self.memory.sample(self.batch_size)
            with torch.no_grad():
                r = torch.from_numpy(np.array([item.reward for item in sampled])).float()
                s_prime = torch.from_numpy(np.array([item.result_state for item in sampled])).float()
                s = torch.from_numpy(np.array([item.current_state for item in sampled])).float()
                a = torch.from_numpy(np.array([item.actions for item in sampled])).long()

            # Traditional `maze-random-5x5-v0` with have a model output a Nx4 output.
            # since r is just Nx1, we spread the reward into the actions.
            y_hat = self.action_model(s).gather(1, a)
            y = self.discount * self.target_net(s_prime).max(axis=1)[0] + r.expand_as(y_hat)

            loss = self.loss_func(y, y_hat)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.action_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
