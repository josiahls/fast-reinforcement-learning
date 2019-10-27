from copy import deepcopy

import numpy as np
import torch
from fastai.basic_train import LearnerCallback, Any, OptimWrapper, ifnone
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam

from fast_rl.agents.agents_base import BaseAgent, get_conv, \
    Flatten
from fast_rl.core.data_block import MDPDataBunch, Action, State
from fast_rl.core.agent_core import ExperienceReplay, OrnsteinUhlenbeck


def get_action_ddpg_cnn(layers, action: Action, state: State, activation=nn.ReLU, kernel_size=5, stride=2):
    module_layers, out_size = get_conv(state.s.shape, activation(), kernel_size=kernel_size, stride=stride,
                                       n_conv_layers=3, layers=[])
    module_layers += [Flatten()]
    layers.append(action.taken_action.shape[1])
    for i, layer in enumerate(layers):
        module_layers += [nn.Linear(out_size, layer)] if i == 0 else [nn.Linear(layers[i - 1], layer)]
        module_layers += [activation()]

    return nn.Sequential(*module_layers)


class BaseDDPGCallback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.max_episodes = 0
        self.episode = 0
        self.iteration = 0
        self.copy_over_frequency = 3

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs

    def on_epoch_begin(self, epoch, **kwargs: Any):
        self.episode = epoch
        self.iteration = 0

    def on_loss_begin(self, **kwargs: Any):
        """Performs memory updates, exploration updates, and model optimization."""
        if self.learn.model.training:
            self.learn.model.memory.update(item=self.learn.data.x.items[-1])
        self.learn.model.exploration_strategy.update(episode=self.episode, max_episodes=self.max_episodes,
                                                     do_exploration=self.learn.model.training)
        post_optimize = self.learn.model.optimize()
        if self.learn.model.training:
            self.learn.model.memory.refresh(post_optimize=post_optimize)
            self.learn.model.target_copy_over()
            self.iteration += 1


class NNActor(nn.Module):
    def __init__(self, layers, action: Action, state: State, activation=nn.ReLU, embed=False):
        super().__init__()
        layers += [action.taken_action.shape[1]]
        module_layers = []

        for i, layer in enumerate(layers):
            module_layers.append(nn.Linear(state.s.shape[1] if i == 0 else layers[i - 1], layer))
            if i != len(layers) - 1: module_layers.append(activation())

        module_layers += [nn.Tanh()]
        self.model = nn.Sequential(*module_layers)

    def forward(self, x):
        return self.model(x)


class CNNActor(nn.Module):
    def __init__(self, layers, action: Action, state: State, activation=nn.ReLU):
        super().__init__()
        # This is still some complete overlap in nn builders, for here, the default function has everything we need
        self.model = get_action_ddpg_cnn(layers, action, state, activation=activation, kernel_size=5, stride=2)

    def forward(self, x):
        return self.model(x)


class NNCritic(nn.Module):
    def __init__(self, layer_list: list, action: Action, state: State):
        super().__init__()
        self.action_size = action.taken_action.shape[1]
        self.state_size = state.s.shape[1]

        self.fc1 = nn.Linear(self.state_size, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0] + self.action_size, layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], 1)

    def forward(self, x):
        x, action = x

        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(torch.cat((x, action), 1)))
        x = nn.LeakyReLU()(self.fc3(x))

        return x


class CNNCritic(nn.Module):
    def __init__(self, action: Action, state: State):
        super().__init__()
        self.action_size = action.taken_action.shape[1]
        self.state_size = state.s.shape

        layers = []
        layers, input_size = get_conv(self.state_size, nn.LeakyReLU(), 8, 2, 3, layers)
        layers += [Flatten()]
        self.conv_layers = nn.Sequential(*layers)

        self.fc1 = nn.Linear(input_size + self.action_size, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        x, action = x

        x = nn.LeakyReLU()(self.conv_layers(x))
        x = nn.LeakyReLU()(self.fc1(torch.cat((x, action), 1)))
        x = nn.LeakyReLU()(self.fc2(x))

        return x


class DDPG(BaseAgent):

    def __init__(self, data: MDPDataBunch, memory=None, tau=1e-3, discount=0.99,
                 lr=1e-3, actor_lr=1e-4, exploration_strategy=None):
        """
        Implementation of a discrete control algorithm using an actor/critic architecture.

        Notes:
            Uses 4 networks, 2 actors, 2 critics.
            All models use batch norm for feature invariance.
            NNCritic simply predicts Q while the Actor proposes the actions to take given a s s.

        References:
            [1] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning."
            arXiv preprint arXiv:1509.02971 (2015).

        Args:
            data: Primary data object to use.
            memory: How big the memory buffer will be for offline training.
            tau: Defines how "soft/hard" we will copy the target networks over to the primary networks.
            discount: Determines the amount of discounting the existing Q reward.
            lr: Rate that the opt will learn parameter gradients.
        """
        super().__init__(data)
        self.name = 'DDPG'
        self.lr = lr
        self.discount = discount
        self.tau = 1
        self.warming_up = True
        self.batch_size = data.train_ds.bs
        self.memory = ifnone(memory, ExperienceReplay(10000))

        self.action_model = self.initialize_action_model([400, 300], data)
        self.critic_model = self.initialize_critic_model([400, 300], data)

        self.opt = OptimWrapper.create(Adam, lr=actor_lr, layer_groups=[self.action_model])
        self.critic_optimizer = OptimWrapper.create(Adam, lr=lr, layer_groups=[self.critic_model])

        self.t_action_model = deepcopy(self.action_model)
        self.t_critic_model = deepcopy(self.critic_model)

        self.target_copy_over()
        self.tau = tau

        self.learner_callbacks = [BaseDDPGCallback]

        self.loss_func = MSELoss()

        self.exploration_strategy = ifnone(exploration_strategy, OrnsteinUhlenbeck(size=data.action.taken_action.shape,
                                                                                   epsilon_start=1, epsilon_end=0.1,
                                                                                   decay=0.001,
                                                                                   do_exploration=self.training))

    def initialize_action_model(self, layers, data):
        if len(data.state.s.shape) == 4 and data.state.s.shape[-1] < 4:
            return CNNActor(layers, data.action, data.state)
        else:
            return NNActor(layers, data.action, data.state)

    def initialize_critic_model(self, layers, data):
        """ Instead of s -> action, we are going s + action -> single expected reward. """
        if len(data.state.s.shape) == 4 and data.state.s.shape[-1] < 4:
            return CNNCritic(data.action, data.state)
        else:
            return NNCritic(layers, data.action, data.state)

    def pick_action(self, x):
        if self.training: self.action_model.eval()
        with torch.no_grad():
            action = super(DDPG, self).pick_action(x)
        if self.training: self.action_model.train()
        return np.clip(action, -1, 1)

    def optimize(self):
        r"""
        Performs separate updates to the actor and critic models.

        Get the predicted yi for optimizing the actor:

        .. math::
                y_i = r_i + \lambda Q^'(s_{i+1}, \; \mu^'(s_{i+1} \;|\; \Theta^{\mu'}}\;|\; \Theta^{Q'})

        On actor optimization, use the actor as the sample policy gradient.

        Returns:

        """
        if len(self.memory) > self.batch_size:
            self.warming_up = False
            # Perhaps have memory as another item list? Should investigate.
            sampled = self.memory.sample(self.batch_size)

            with torch.no_grad():
                r = torch.cat([item.reward.float() for item in sampled])
                s_prime = torch.cat([item.s_prime.float() for item in sampled])
                s = torch.cat([item.s.float() for item in sampled])
                a = torch.cat([item.a.float() for item in sampled])
                # d = torch.cat([item.done.float() for item in sampled]) # Do we need a mask??

            with torch.no_grad():
                y = r + self.discount * self.t_critic_model((s_prime, self.t_action_model(s_prime)))

            y_hat = self.critic_model((s, a))

            critic_loss = self.loss_func(y_hat, y)

            if self.training:
                # Optimize critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            actor_loss = -self.critic_model((s, self.action_model(s))).mean()

            self.loss = critic_loss.cpu().detach()

            if self.training:
                # Optimize actor network
                self.opt.zero_grad()
                actor_loss.backward()
                self.opt.step()

            with torch.no_grad():
                post_info = {'td_error': (y - y_hat).numpy()}
                return post_info

    def forward(self, x):
        x = super(DDPG, self).forward(x)
        return self.action_model(x)

    def target_copy_over(self):
        """ Soft target updates the actor and critic models.."""
        self.soft_target_copy_over(self.t_action_model, self.action_model, self.tau)
        self.soft_target_copy_over(self.t_critic_model, self.critic_model, self.tau)

    def soft_target_copy_over(self, t_m, f_m, tau):
        for target_param, local_param in zip(t_m.parameters(), f_m.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def interpret_q(self, items):
        with torch.no_grad():
            r = torch.from_numpy(np.array([item.reward for item in items])).float()
            s_prime = torch.from_numpy(np.array([item.result_state for item in items])).float()
            s = torch.from_numpy(np.array([item.current_state for item in items])).float()
            a = torch.from_numpy(np.array([item.actions for item in items])).float()

            return self.critic_model(torch.cat((s, a), 1))
