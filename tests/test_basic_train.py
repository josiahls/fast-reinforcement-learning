import os



from fast_rl.agents.dqn import create_dqn_model, FixedTargetDQNModule, dqn_learner
from fast_rl.core.agent_core import ExperienceReplay, torch, GreedyEpsilon
from fast_rl.core.basic_train import load_learner
from fast_rl.core.data_block import MDPDataBunch


def test_fit():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20, add_valid=False)
	model=create_dqn_model(data, FixedTargetDQNModule, opt=torch.optim.RMSprop)
	memory=ExperienceReplay(memory_size=1000, reduce_ram=True)
	exploration_method=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner=dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
	learner.fit(2)
	learner.fit(2)
	learner.fit(2)

	assert len(data.x.info)==6
	assert 0 in data.x.info
	assert 5 in data.x.info


def test_to_pickle():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20, add_valid=False)
	model=create_dqn_model(data, FixedTargetDQNModule, opt=torch.optim.RMSprop)
	memory=ExperienceReplay(memory_size=1000, reduce_ram=True)
	exploration_method=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner=dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
	learner.fit(2)

	assert len(data.x.info)==2
	assert 0 in data.x.info
	assert 1 in data.x.info

	data.to_pickle('./data/test_to_pickle')
	assert os.path.exists('./data/test_to_pickle_CartPole-v0')


def test_from_pickle():
	data=MDPDataBunch.from_pickle('./data/test_to_pickle_CartPole-v0')
	model=create_dqn_model(data, FixedTargetDQNModule, opt=torch.optim.RMSprop)
	memory=ExperienceReplay(memory_size=1000, reduce_ram=True)
	exploration_method=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner=dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
	learner.fit(2)

	assert len(data.x.info)==4
	assert 0 in data.x.info
	assert 3 in data.x.info


def test_export_learner():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20, add_valid=False)
	model=create_dqn_model(data, FixedTargetDQNModule, opt=torch.optim.RMSprop)
	memory=ExperienceReplay(memory_size=1000, reduce_ram=True)
	exploration_method=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner=dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
	learner.fit(2)

	learner.export('test_export.pkl')#, pickle_data=True)
	learner = load_learner(learner.path, 'test_export.pkl')
	learner.fit(2)
