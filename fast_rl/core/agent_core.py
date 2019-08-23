import collections
import heapq
import math
import random
import numpy as np
from collections import deque

from typing import List

import gym
from torch import nn

from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice


class ExplorationStrategy:
    def __init__(self, do_exploration: bool):
        self.do_exploration = do_exploration

    def perturb(self, action, action_space):
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

    def update(self, do_exploration, **kwargs):
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
        TODO for now does random discrete selection. Move to continuous soon.

        Args:
            action:
            action_space:

        Returns:
        """
        if np.random.random() < self.epsilon:
            return action_space.sample()
        else:
            return action

    def update(self, current_episode, end_episode=0, **kwargs):
        super(GreedyEpsilon, self).update(**kwargs)
        self.end_episode = end_episode
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                           math.exp(-1. * (self.steps_done * self.decay))
        self.steps_done += 1


class Experience:
    def __init__(self, memory_size):
        self.max_size = memory_size

    def sample(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass

    def refresh(self, **kwargs):
        pass


class ExperienceReplay(Experience):
    def __init__(self, memory_size):
        """
        Basic store-er of state space transitions for training agents.

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).

        Args:
            memory_size (int): Max N samples to store
        """
        super().__init__(memory_size)
        self.max_size = memory_size
        self.memory = deque(maxlen=memory_size)  # type: List[MarkovDecisionProcessSlice]

    def __len__(self):
        return len(self.memory)

    def sample(self, batch, **kwargs):
        if len(self.memory) < batch: return self.memory
        return random.sample(self.memory, batch)

    def update(self, item, **kwargs):
        self.memory.append(item)


class PriorityExperienceReplay(Experience):
    def __init__(self, memory_size, batch_size=64, epsilon=0.001, alpha=1):
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
        super().__init__(memory_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.memory = []  # type: np.array([PriorityItem])
        # When sampled, store the sample indices for refresh.
        self._indices = np.array([])  # type: List[int]

    def refresh(self, td_error, **kwargs):
        np.array(self.memory)[self._indices] = td_error

    def sample(self, **kwargs):
        self._indices = [np.random.randint(0, len(self.memory)) for _ in range(self.batch_size)]
        return np.array(self.memory)[self._indices]

    def update(self, item, **kwargs):
        """
        Updates the memory of PER.

        If the memory is at its max, we ensure that the inserted item has the max possible priority so that it
        can be sampled and ultimately given a proper priority value.

        Args:
            item:

        Returns:

        """
        if len(self.memory) < self.max_size: heapq.heappush(self.memory, PriorityItem(item, 1 + self.epsilon))
        else: heapq.heappushpop(self.memory, PriorityItem(item, self.memory[0] + self.epsilon))












