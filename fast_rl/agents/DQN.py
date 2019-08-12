import torch
from fastai.basic_train import LearnerCallback, Any
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


class DQN(BaseAgent):
    def __init__(self, data: MDPDataBunch):
        """Trains an Agent using the Q Learning method on a neural net.

        Args:
            data:

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).
        """
        super().__init__(data)
        # TODO add recommend cnn based on state size?
        self.batch_size = 100
        self.discount = 0.1
        self.lr = 0.1
        self.loss_func = MSELoss()
        self.memory = ExperienceReplay(500)
        self.action_model = create_nn_model([10, 10, 10], *data.get_action_state_size())
        self.optimizer = optim.Adam(self.action_model.parameters(), lr=self.lr)
        self.callbacks += [BaseDQNCallback(self)]
        self.exploration_strategy = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.1,
                                                  do_exploration=self.training)

    def forward(self, x):
        x = super(DQN, self).forward(x)
        return self.pick_action(self.action_model(x))

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
            y_hat = self.action_model(s_prime).gather(1, a)
            y = self.discount * self.action_model(s).gather(1, a) + r.expand_as(y_hat)

            loss = self.loss_func(y, y_hat)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()




