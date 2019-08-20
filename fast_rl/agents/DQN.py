from copy import deepcopy

import torch
from fastai.basic_train import LearnerCallback, Any, F
from torch import optim, nn
from torch.nn import MSELoss
import numpy as np

from fast_rl.agents.BaseAgent import BaseAgent, create_nn_model
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon


class BaseDQNCallback(LearnerCallback):
    def __init__(self, learn):
        """Handles basic DQN end of step model optimization."""
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
        """Performs memory updates, exploration updates, and model optimization."""
        learn.model.memory.update(item=learn.data.x.items[-1])
        learn.model.exploration_strategy.update(self.episode, self.max_episodes, do_exploration=learn.model.training)
        post_optimize = learn.model.optimize()
        learn.model.memory.refresh(post_optimize)
        self.iteration += 1


class FixedTargetDQNCallback(BaseDQNCallback):
    def __init__(self, learn, copy_over_frequency=3):
        """
        Handles updating the target model in a fixed target DQN.

        Args:
            learn: Basic Learner.
            copy_over_frequency: Per how many episodes we want to update the target model.
        """
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

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).

        Args:
            data: Used for size input / output information.
        """
        super().__init__(data)
        # TODO add recommend cnn based on state size?
        self.batch_size = 64
        self.discount = 0.99
        self.lr = 0.001
        self.loss_func = F.smooth_l1_loss
        self.memory = ExperienceReplay(1000)
        self.action_model = self.initialize_action_model([10, 10, 10], data)
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=self.lr)
        self.callbacks += [BaseDQNCallback(self)]
        self.exploration_strategy = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001,
                                                  do_exploration=self.training)

    def initialize_action_model(self, layers, data):
        return create_nn_model(layers, *data.get_action_state_size())

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
            for param in self.action_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            post_info = {'td_error', y - y_hat}
            return post_info


class FixedTargetDQN(DQN):
    def __init__(self, data: MDPDataBunch):
        """Trains an Agent using the Q Learning method on a 2 neural nets.

        Notes:
            Unlike the base DQN, this is a true reflection of ref [1]. We use 2 models instead of one to allow for
            training the action model more stably.

        Args:
            data: Used for size input / output information.

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).
        """
        super().__init__(data)
        self.target_net = deepcopy(self.action_model)
        self.callbacks += [FixedTargetDQNCallback(self, 10)]

    def target_copy_over(self):
        """ Updates the target network from calls in the FixedTargetDQNCallback callback."""
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
            # Perhaps have memory as another item list? Should investigate.
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

            post_info = {'td_error', y - y_hat}
            return post_info


class DoubleDQN(FixedTargetDQN):
    def __init__(self, data: MDPDataBunch):
        """
        Double DQN training.

        References:
            [1] Van Hasselt, Hado, Arthur Guez, and David Silver. "Deep reinforcement learning with double q-learning."
            Thirtieth AAAI conference on artificial intelligence. 2016.

        Args:
            data: Used for size input / output information.
        """
        super().__init__(data)

    def optimize(self):
        """
        Uses ER to optimize the Q-net.

        Uses the equation:

        .. math::
                Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{}(Q^{*}(s' , \
                argmax_{a'}(Q(s', \Theta)), \Theta^{-})) \;|\; s, a \Big]

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
            y = self.discount * self.target_net(s_prime).gather(1, self.action_model(s_prime).argmax(axis=1).unsqueeze(
                1)) + r.expand_as(y_hat)

            loss = self.loss_func(y, y_hat)

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.action_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()


class DuelingDQNModule(nn.Module):
    def __init__(self, layers, data):
        super().__init__()
        self.base = create_nn_model(layers, *data.get_action_state_size())[:-1]
        a_s = data.get_action_state_size()
        stream_input_size = self.base[-2].out_features

        self.val = create_nn_model([stream_input_size], 1, stream_input_size)
        self.adv = create_nn_model([stream_input_size], a_s[0], stream_input_size)

    def forward(self, x):
        """Splits the base neural net output into 2 streams to evaluate the advantage and values of the state space and
        corresponding actions.

        .. math::
           Q(s,a;\; \Theta, \\alpha, \\beta) = V(s;\; \Theta, \\beta) + A(s, a;\; \Theta, \\alpha) - \\frac{1}{|A|}
           \\Big\\sum_{a'} A(s, a';\; \Theta, \\alpha)

        Args:
            x:

        Returns:

        """
        x = self.base(x)
        val = self.val(x)
        adv = self.adv(x)

        x = val.expand_as(adv) + (adv - adv.mean())
        return x


class DuelingDQN(FixedTargetDQN):
    def __init__(self, data: MDPDataBunch):
        """Replaces the basic action model with a DuelingDQNModule which splits the basic model into 2 streams.


        References:
            [1] Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning."
            arXiv preprint arXiv:1511.06581 (2015).

        Args:
            data:
        """
        super().__init__(data)

    def initialize_action_model(self, layers, data):
        return DuelingDQNModule(data=data, layers=layers)


class DoubleDuelingDQN(DoubleDQN, DuelingDQN):
    def __init__(self, data: MDPDataBunch):
        """
        Combines both Dueling DQN and DDQN.

        Args:
            data: Used for size input / output information.
        """
        super().__init__(data)
