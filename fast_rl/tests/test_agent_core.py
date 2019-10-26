import pytest

from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import DQN, FixedTargetDQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import PriorityExperienceReplay
from fast_rl.core.basic_train import AgentLearner


def test_priority_experience_replay():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(3)
    data.train_ds.env.close()


@pytest.mark.parametrize("env", sorted(['CartPole-v0']))
def test_databunch_dqn_fit(env):
    data = MDPDataBunch.from_env(env)
    model = DQN(data)
    learner = AgentLearner(data=data, model=model)
    learner.fit(3)
    data.valid_ds.env.close()
    data.train_ds.env.close()

def test_fit_function_ddpg():
    data = MDPDataBunch.from_env('Pendulum-v0', bs=4, render='human', max_steps=100, add_valid=False)
    model = DDPG(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(3)
    data.train_ds.env.close()


