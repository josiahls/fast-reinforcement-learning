from fastai.imports import torch

from fast_rl.agents.dqn import create_dqn_model, dqn_learner
from fast_rl.agents.dqn_models import DQNModule
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.metrics import RewardMetric, EpsilonMetric


def test_metrics_reward_init():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20)
	model=create_dqn_model(data, DQNModule, opt=torch.optim.RMSprop)
	memory=ExperienceReplay(memory_size=1000, reduce_ram=True)
	exploration_method=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner=dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
						callback_fns=[RewardMetric])
	learner.fit(2)


def test_metrics_epsilon_init():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20)
	model=create_dqn_model(data, DQNModule, opt=torch.optim.RMSprop)
	memory=ExperienceReplay(memory_size=1000, reduce_ram=True)
	exploration_method=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner=dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
						callback_fns=[EpsilonMetric])
	learner.fit(2)

