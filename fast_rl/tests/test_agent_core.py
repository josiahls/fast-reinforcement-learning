from fastai.basic_train import LearnerCallback, DatasetType
from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import PriorityExperienceReplay


def test_priority_experience_replay():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = FixedTargetDQN(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(5)


def test_fit_function_ddpg():
    data = MDPDataBunch.from_env('Pendulum-v0', render='human', max_steps=1000)
    model = DDPG(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(5)


def test_fit_function_dqn():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DQN(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(5)
