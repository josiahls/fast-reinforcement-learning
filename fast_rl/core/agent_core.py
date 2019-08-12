import collections
import random
import numpy as np
from collections import deque

from typing import List

import gym

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
        self.epsilon = self.epsilon / (1.0 + (current_episode / self.decay))


class Experience:
    def sample(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class ExperienceReplay(Experience):
    def __init__(self, memory_size):
        self.max_size = memory_size
        self.memory = deque(maxlen=memory_size)  # type: List[MarkovDecisionProcessSlice]

    def __len__(self):
        return len(self.memory)

    def sample(self, batch, **kwargs):
        if len(self.memory) < batch: return self.memory
        return random.sample(self.memory, batch)

    def update(self, item, **kwargs):
        self.memory.append(item)
