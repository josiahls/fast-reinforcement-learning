import pytest
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
from fast_rl.core.Learner import AgentLearnerAlpha, AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunchAlpha, FEED_TYPE_STATE, MDPDataBunch
from fast_rl.core.agent_core import PriorityExperienceReplay, ExperienceReplay


def test_priority_experience_replay():
    data = MDPDataBunchAlpha.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearnerAlpha(data, model)
    learn.fit(5)


@pytest.mark.parametrize("env", sorted(['CartPole-v0']))
def test_databunch_dqn_fit(env):
    data = MDPDataBunch.from_env(env)
    model = DQN(data)
    learner = AgentLearner(data=data, model=model)

    learner.fit(5)



# def test_fit_function_ddpg():
#     data = MDPDataBunchAlpha.from_env('Pendulum-v0', render='human', max_steps=100, add_valid=False)
#     model = DDPG(data, memory=PriorityExperienceReplay(1000))
#     learn = AgentLearnerAlpha(data, model)
#     learn.fit(5)


