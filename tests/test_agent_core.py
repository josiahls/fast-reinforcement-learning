import pytest

from fast_rl.agents.ddpg import DDPG
from fast_rl.agents.dqn import DQN, FixedTargetDQN
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.agent_core import PriorityExperienceReplay
from fast_rl.core.basic_train import AgentLearner


def test_priority_experience_replay():
    data = MDPDataBunch.from_env('CartPole-v0', render='human', bs=4, max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(3)


def test_agent_recorder_loss_plot():
    data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array')
    model = DQN(data)
    learn = AgentLearner(data, model)
    learn.fit(1)
    # learn.recorder.plot()
    # learn.recorder.plot_losses()


def test_databunch_dqn_fit():
    data = MDPDataBunch.from_env('CartPole-v0')
    model = DQN(data)
    learner = AgentLearner(data=data, model=model)
    learner.fit(3)


def test_fit_function_ddpg():
    data = MDPDataBunch.from_env('Pendulum-v0', bs=4, render='human', max_steps=20, add_valid=False)
    model = DDPG(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(3)

