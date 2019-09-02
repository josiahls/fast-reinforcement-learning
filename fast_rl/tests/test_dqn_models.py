from fastai.basic_train import LearnerCallback, DatasetType
from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


def test_basic_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
    model = DQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_fixed_target_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = FixedTargetDQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_double_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DoubleDQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_dueling_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DuelingDQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_double_dueling_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DoubleDuelingDQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)
