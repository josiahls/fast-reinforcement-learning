import collections
import random
from collections import deque

from typing import List

from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice


class Experience:
    def sample(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass


class ExperienceReplay(Experience):
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)  # type: List[MarkovDecisionProcessSlice]

    def sample(self, batch, **kwargs):
        if len(self.memory) < batch: return self.memory
        return random.sample(self.memory, batch)

    def update(self, item, **kwargs):
        self.memory.append(item)
