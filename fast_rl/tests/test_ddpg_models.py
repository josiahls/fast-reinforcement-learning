from collections import Collection

from fastai.basic_train import LearnerCallback

from fast_rl.agents.DDPG import DDPG
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


def test_ddpg():
    data = MDPDataBunch.from_env('Pendulum-v0', render='human')
    # data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human')
    model = DDPG(data, batch=8)
    learn = AgentLearner(data, model)
    learn.fit(450)