import torch
from fastai.basic_train import LearnerCallback, Any
import numpy as np
from fastai.metrics import RMSE
from torch.nn import MSELoss
from torch.optim import Adam

from fast_rl.agents.BaseAgent import BaseAgent, create_nn_model
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import GreedyEpsilon, ExperienceReplay


class BaseDDPGCallback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.max_episodes = 0
        self.episode = 0
        self.iteration = 0
        self.copy_over_frequency = 2

    def on_train_begin(self, learn: AgentLearner, max_episodes, **kwargs: Any):
        self.max_episodes = max_episodes

    def on_epoch_begin(self, episode, **kwargs: Any):
        self.episode = episode
        self.iteration = 0

    def on_step_end(self, learn: AgentLearner, **kwargs: Any):
        """Performs memory updates, exploration updates, and model optimization."""
        learn.model.memory.update(item=learn.data.x.items[-1])
        learn.model.exploration_strategy.update(self.episode, self.max_episodes, do_exploration=learn.model.training)
        post_optimize = learn.model.optimize()
        learn.model.memory.refresh(post_optimize=post_optimize)
        self.iteration += 1

    def on_epoch_end(self, learn: AgentLearner, **kwargs: Any):
        if self.episode % self.copy_over_frequency == 0:
            learn.model.target_copy_over()


class DDPG(BaseAgent):

    def __init__(self, data: MDPDataBunch, memory=ExperienceReplay(10000), tau=0.001, batch=64, discount=0.99, lr=0.005):
        """
        Implementation of a continuous control algorithm using an actor/critic architecture.

        Notes:
            Uses 4 networks, 2 actors, 2 critics.
            All models use batch norm for feature invariance.
            Critic simply predicts Q while the Actor proposes the actions to take given a state s.

        References:
            [1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning."
            arXiv preprint arXiv:1509.02971 (2015).

        Args:
            data: Primary data object to use.
            memory: How big the memory buffer will be for offline training.
            tau: Defines how "soft/hard" we will copy the target networks over to the primary networks.
            batch: Size of per memory query.
            discount: Determines the amount of discounting the existing Q reward.
            lr: Rate that the optimizer will learn parameter gradients.
        """
        super().__init__(data)
        self.lr = lr
        self.discount = discount
        self.batch = batch
        self.tao = tau
        self.memory = memory

        self.action_model = self.initialize_action_model([64, 64], data)
        self.critic_model = self.initialize_critic_model([64, 64], data)

        self.actor_optimizer = Adam(self.action_model.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=lr)

        self.t_action_model = self.initialize_action_model([64, 64], data)
        self.t_critic_model = self.initialize_critic_model([64, 64], data)

        self.target_copy_over()

        self.callbacks = [BaseDDPGCallback(self)]

        self.loss_func = MSELoss()
        # TODO Move to Ornstein-Uhlenbeck process
        self.exploration_strategy = GreedyEpsilon(decay=0.001, epsilon_end=0.1, epsilon_start=1, do_exploration=True)

    def initialize_action_model(self, layers, data):
        return create_nn_model(layers, *data.get_action_state_size(), True)

    def initialize_critic_model(self, layers, data):
        """ Instead of state -> action, we ware going state + action -> single expected reward. """
        return create_nn_model(layers, 1, sum(data.get_action_state_size()), True)

    def pick_action(self, x):
        if self.training: self.action_model.eval()
        with torch.no_grad():
            action = super(DDPG, self).pick_action(x)
        if self.training: self.action_model.train()
        return action

    def optimize(self):
        """
        Performs separate updates to the actor and critic models.

        Get the predicted yi for optimizing the actor:

        .. math::
                y_i = r_i + \lambda Q^'(s_{i+1}, \; \mu^'(s_{i+1} \;|\; \Theta^{\mu'}}\;|\; \Theta^{Q'})

        On actor optimization, use the actor as the sample policy gradient.

        Returns:

        """
        if len(self.memory) > self.batch:
            # Perhaps have memory as another item list? Should investigate.
            sampled = self.memory.sample(self.batch)
            with torch.no_grad():
                r = torch.from_numpy(np.array([item.reward for item in sampled]).astype(float)).float()
                s_prime = torch.from_numpy(np.array([item.result_state for item in sampled])).float()
                s = torch.from_numpy(np.array([item.current_state for item in sampled])).float()
                a = torch.from_numpy(np.array([item.actions for item in sampled]).astype(float)).float()

            with torch.no_grad():
                y = r + self.discount * self.t_critic_model(torch.cat((s_prime, self.t_action_model(s_prime)), 1))

            y_hat = self.critic_model(torch.cat((s, a), 1))

            critic_loss = self.loss_func(y, y_hat)

            # Optimize critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            for param in self.critic_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.critic_optimizer.step()

            actor_loss = -self.critic_model(torch.cat((s, self.action_model(s)), 1)).mean()

            # Optimize actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for param in self.action_model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.actor_optimizer.step()

    def forward(self, x):
        x = super(DDPG, self).forward(x)
        return self.action_model(x)

    def target_copy_over(self):
        """ Soft target updates the actor and critic models.."""
        self.soft_target_copy_over(self.t_action_model, self.action_model, self.tao)
        self.soft_target_copy_over(self.t_critic_model, self.critic_model, self.tao)

    def soft_target_copy_over(self, t_m, f_m, tau):
        for target_param, local_param in zip(t_m.parameters(), f_m.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def interpret_q(self, items):
        with torch.no_grad():
            r = torch.from_numpy(np.array([item.reward for item in items])).float()
            s_prime = torch.from_numpy(np.array([item.result_state for item in items])).float()
            s = torch.from_numpy(np.array([item.current_state for item in items])).float()
            a = torch.from_numpy(np.array([item.actions for item in items])).float()

            return self.critic_model(torch.cat((s, a), 1))
