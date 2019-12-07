import copy
from collections import deque
from math import ceil

import gym
from fastai.basic_train import *
from fastai.torch_core import *

from fast_rl.core.data_structures import SumTree


class ExplorationStrategy:
    def __init__(self, do_exploration: bool=True):
        self.do_exploration = do_exploration

    def perturb(self, action,  action_space):
        """
        Base method just returns the action. Subclass, and change to return randomly / augmented actions.

        Should use `do_exploration` field. It is recommended that when you subclass / overload, you allow this field
        to completely bypass these actions.

        Args:
            action:
            action_space (gym.Space): The original gym space. Should contain information on the action type, and
            possible convenience methods for random action selection.

        Returns:

        """
        _ = action_space
        return action

    def update(self, max_episodes, do_exploration, **kwargs):
        self.do_exploration = do_exploration


class GreedyEpsilon(ExplorationStrategy):
    def __init__(self, epsilon_start, epsilon_end, decay, start_episode=0, end_episode=0, **kwargs):
        super().__init__(**kwargs)
        self.end_episode = end_episode
        self.start_episode = start_episode
        self.decay = decay
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.steps_done = 0

    def perturb(self, action, action_space: gym.Space):
        """
        TODO for now does random discrete selection. Move to discrete soon.

        Args:
            action:
            action_space:

        Returns:
        """
        if np.random.random() < self.epsilon:
            return action_space.sample()
        else:
            return action

    def update(self, episode, end_episode=0, **kwargs):
        super(GreedyEpsilon, self).update(**kwargs)
        if self.do_exploration:
            self.end_episode = end_episode
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           math.exp(-1. * (self.steps_done * self.decay))
            self.steps_done += 1


class OrnsteinUhlenbeck(GreedyEpsilon):
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2, **kwargs):
        """

        References:
            [1] From https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
            [2] Cumulatively based on

        Args:
            epsilon_start:
            epsilon_end:
            decay:
            **kwargs:
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.theta = theta
        self.mu = mu
        self.x = np.ones(size)

    def perturb(self, action, action_space):
        if self.do_exploration:
            dx = self.theta * (self.mu - self.x) + self.sigma * np.array([np.random.normal() for _ in range(len(self.x))])
        else: dx = np.zeros(self.x.shape)

        self.x += dx
        return self.epsilon * self.x + action


class Experience:
    def __init__(self, memory_size, reduce_ram=False):
        self.reduce_ram = reduce_ram
        self.max_size = memory_size
        self.callbacks = []

    def sample(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass

    def refresh(self, **kwargs):
        pass


class ExperienceReplay(Experience):
    def __init__(self, memory_size, **kwargs):
        r"""
        Basic store-er of s space transitions for training agents.

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).

        Args:
            memory_size (int): Max N samples to store
        """
        super().__init__(memory_size, **kwargs)
        self.max_size = memory_size
        self.memory = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.memory)

    def sample(self, batch, **kwargs):
        if len(self.memory) < batch: return self.memory
        return random.sample(self.memory, batch)

    def update(self, item, **kwargs):
        if self.reduce_ram: item.clean()
        self.memory.append(deepcopy(item))


class PriorityExperienceReplayCallback(LearnerCallback):
    def on_train_begin(self, **kwargs):
        self.learn.model.loss_func = partial(self.learn.model.memory.handle_loss,
                                             base_function=self.learn.model.loss_func,
                                             device=self.learn.data.device)


class PriorityExperienceReplay(Experience):

    def handle_loss(self, y, y_hat, base_function, device):
        return (base_function(y, y_hat) * torch.from_numpy(self.priority_weights).to(device=device).float()).mean()

    def __init__(self, memory_size, batch_size=64, epsilon=0.01, alpha=0.6, beta=0.4, b_inc=-0.001, **kwargs):
        """
        Prioritizes sampling based on samples requiring the most learning.

        References:
            [1] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

        Args:
            batch_size (int): Size of sample, and thus size of expected index update.
            alpha (float): Changes the sampling behavior 1 (non-uniform) -> 0 (uniform)
            epsilon (float): Keeps the probabilities of items from being 0
            memory_size (int): Max N samples to store
        """
        super().__init__(memory_size, **kwargs)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.b_inc = b_inc
        self.priority_weights = None  # np.zeros(self.batch_size, dtype=float)
        self.epsilon = epsilon
        self.memory = SumTree(self.max_size)
        self.callbacks = [PriorityExperienceReplayCallback]
        # When sampled, store the sample indices for refresh.
        self._indices = None  # np.zeros(self.batch_size, dtype=int)

    def __len__(self):
        return self.memory.n_entries

    def refresh(self, post_optimize, **kwargs):
        if post_optimize is not None:
            self.memory.update(self._indices.astype(int), np.abs(post_optimize['td_error']) + self.epsilon)

    def sample(self, batch, **kwargs):
        self.beta = np.min([1., self.beta + self.b_inc])
        # ranges = np.linspace(0, self.memory.total(), num=ceil(self.memory.total() / self.batch_size))
        ranges = np.linspace(0, ceil(self.memory.total() / batch), num=batch + 1)
        uniform_ranges = [np.random.uniform(ranges[i], ranges[i + 1]) for i in range(len(ranges) - 1)]
        self._indices, weights, samples = self.memory.batch_get(uniform_ranges)
        self.priority_weights = self.memory.anneal_weights(weights, self.beta)
        return samples

    def update(self, item, **kwargs):
        """
        Updates the memory of PER.

        Assigns maximal priority per [1] Alg:1, thus guaranteeing that sample being visited once.

        Args:
            item:

        Returns:

        """
        maximal_priority = self.alpha
        if self.reduce_ram: item.clean()
        self.memory.add(np.abs(maximal_priority) + self.epsilon, item)


class HindsightExperienceReplay(Experience):
    def __init__(self, memory_size):
        """

        References:
            [1] Andrychowicz, Marcin, et al. "Hindsight experience replay."
            Advances in Neural Information Processing Systems. 2017.

        Args:
            memory_size:
        """
        super().__init__(memory_size)
