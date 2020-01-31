from itertools import product
from time import sleep

import pytest
from fastai.basic_data import DatasetType

from fast_rl.agents.dqn import create_dqn_model, dqn_learner, DQNLearner
from fast_rl.agents.dqn_models import *
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, GreedyEpsilon
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_STATE, FEED_TYPE_IMAGE, ResolutionWrapper
from fast_rl.core.metrics import RewardMetric, EpsilonMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation
from torch import optim

p_model = [DQNModule, FixedTargetDQNModule,DoubleDuelingModule,DuelingDQNModule,DoubleDQNModule]
p_exp = [ExperienceReplay,
		PriorityExperienceReplay]
p_format = [FEED_TYPE_STATE]#, FEED_TYPE_IMAGE]
p_envs = ['CartPole-v1']

config_env_expectations = {
	'CartPole-v1': {'action_shape': (1, 2), 'state_shape': (1, 4)},
	'maze-random-5x5-v0': {'action_shape': (1, 4), 'state_shape': (1, 2)}
}


def learner2gif(lnr:DQNLearner,s_format,group_interp:GroupAgentInterpretation,name:str,extra:str):
	meta=f'{lnr.memory.__class__.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
	interp=AgentInterpretation(lnr, ds_type=DatasetType.Train)
	interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
	group_interp.add_interpretation(interp)
	group_interp.to_pickle(f'../docs_src/data/{name}_{lnr.model.name.lower()}/', f'{lnr.model.name.lower()}_{meta}')
	temp=[g.write(f'../res/run_gifs/{name}_{extra}') for g in interp.generate_gif()]
	del temp
	gc.collect()



def trained_learner(model_cls, env, s_format, experience, bs, layers, memory_size=1000000, decay=0.001,
					copy_over_frequency=300, lr=None, epochs=450,**kwargs):
	if lr is None: lr = [0.001, 0.00025]
	memory = experience(memory_size=memory_size, reduce_ram=True)
	explore = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=decay)
	if type(lr) == list: lr = lr[0] if model_cls == DQNModule else lr[1]
	data = MDPDataBunch.from_env(env, render='human', bs=bs, add_valid=False, keep_env_open=False, feed_type=s_format,
								 memory_management_strategy='k_partitions_top', k=3,**kwargs)
	if model_cls == DQNModule: model = create_dqn_model(data=data, base_arch=model_cls, lr=lr, layers=layers, opt=optim.RMSprop)
	else: model = create_dqn_model(data=data, base_arch=model_cls, lr=lr, layers=layers)
	learn = dqn_learner(data, model, memory=memory, exploration_method=explore, copy_over_frequency=copy_over_frequency,
						callback_fns=[RewardMetric, EpsilonMetric])
	learn.fit(epochs)
	return learn

# @pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "env"], list(product(p_model, p_format, p_envs)))
def test_dqn_create_dqn_model(model_cls, s_format, env):
	data = MDPDataBunch.from_env(env, render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
	model = create_dqn_model(data, model_cls)
	model.eval()
	model(data.state.s)

	assert config_env_expectations[env]['action_shape'] == (1, data.action.n_possible_values.item())
	if s_format == FEED_TYPE_STATE:
		assert config_env_expectations[env]['state_shape'] == data.state.s.shape


# @pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "mem", "env"], list(product(p_model, p_format, p_exp, p_envs)))
def test_dqn_dqn_learner(model_cls, s_format, mem, env):
	data = MDPDataBunch.from_env(env, render='rgb_array', bs=32, add_valid=False, keep_env_open=False, feed_type=s_format)
	model = create_dqn_model(data, model_cls)
	memory = mem(memory_size=1000, reduce_ram=True)
	exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)

	assert config_env_expectations[env]['action_shape'] == (1, data.action.n_possible_values.item())
	if s_format == FEED_TYPE_STATE:
		assert config_env_expectations[env]['state_shape'] == data.state.s.shape


# @pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "mem", "env"], list(product(p_model, p_format, p_exp, p_envs)))
def test_dqn_fit(model_cls, s_format, mem, env):
	data = MDPDataBunch.from_env(env, render='rgb_array', bs=5, max_steps=20, add_valid=False, keep_env_open=False, feed_type=s_format)
	model = create_dqn_model(data, model_cls, opt=torch.optim.RMSprop)
	memory = mem(memory_size=1000, reduce_ram=True)
	exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
	learner.fit(2)

	assert config_env_expectations[env]['action_shape'] == (1, data.action.n_possible_values.item())
	if s_format == FEED_TYPE_STATE:
		assert config_env_expectations[env]['state_shape'] == data.state.s.shape


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "mem"], list(product(p_model, p_format, p_exp)))
def test_dqn_fit_maze_env(model_cls, s_format, mem):
	group_interp = GroupAgentInterpretation()
	extra_s=f'{mem.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		learn = trained_learner(model_cls, 'maze-random-5x5-v0', s_format, mem, bs=32, layers=[32, 32],
								memory_size=1000000, decay=0.00001, res_wrap=partial(ResolutionWrapper, w_step=3, h_step=3))

		learner2gif(learn,s_format,group_interp,'maze_5x5',extra_s)
	# success = False
	# while not success:
	# 	try:
	# 		data = MDPDataBunch.from_env('maze-random-5x5-v0', render='rgb_array', bs=5, max_steps=20,
	# 									 add_valid=False, keep_env_open=False, feed_type=s_format)
	# 		model = create_dqn_model(data, model_cls, opt=torch.optim.RMSprop)
	# 		memory = ExperienceReplay(10000)
	# 		exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	# 		learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
	# 							  callback_fns=[RewardMetric, EpsilonMetric])
	# 		learner.fit(2)
	#
	# 		assert config_env_expectations['maze-random-5x5-v0']['action_shape'] == (
	# 			1, data.action.n_possible_values.item())
	# 		if s_format == FEED_TYPE_STATE:
	# 			assert config_env_expectations['maze-random-5x5-v0']['state_shape'] == data.state.s.shape
	# 		sleep(1)
	# 		success = True
	# 	except Exception as e:
	# 		if not str(e).__contains__('Surface'):
	# 			raise Exception


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'], list(product(p_model, p_format, p_exp)))
def test_dqn_models_minigrids(model_cls, s_format, experience):
	group_interp = GroupAgentInterpretation()
	for i in range(5):
		learn = trained_learner(model_cls, 'MiniGrid-FourRooms-v0', s_format, experience, bs=32, layers=[64, 64],
								memory_size=1000000, decay=0.00001, epochs=1000)

		meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
		interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		group_interp.add_interpretation(interp)
		filename = f'{learn.model.name.lower()}_{meta}'
		group_interp.to_pickle(f'../docs_src/data/minigrid_{learn.model.name.lower()}/', filename)
		[g.write('../res/run_gifs/minigrid') for g in interp.generate_gif()]
		del learn


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
						 list(product(p_model, p_format, p_exp)))
def test_dqn_models_cartpole(model_cls, s_format, experience):
	group_interp = GroupAgentInterpretation()
	extra_s=f'{experience.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		learn = trained_learner(model_cls, 'CartPole-v1', s_format, experience, bs=32, layers=[64, 64],
								memory_size=1000000, decay=0.001)

		learner2gif(learn,s_format,group_interp,'cartpole',extra_s)
		# meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		# interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
		# interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		# group_interp.add_interpretation(interp)
		# filename = f'{learn.model.name.lower()}_{meta}'
		# group_interp.to_pickle(f'../docs_src/data/cartpole_{learn.model.name.lower()}/', filename)
		# [g.write('../res/run_gifs/cartpole') for g in interp.generate_gif()]
		# del learn


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'], list(product(p_model, p_format, p_exp)))
def test_dqn_models_lunarlander(model_cls, s_format, experience):
	group_interp = GroupAgentInterpretation()
	extra_s=f'{experience.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		learn = trained_learner(model_cls, 'LunarLander-v2', s_format, experience, bs=32, layers=[128, 64],
								memory_size=1000000, decay=0.00001, copy_over_frequency=600, lr=[0.001, 0.00025],
								epochs=1000)
		learner2gif(learn, s_format, group_interp, 'lunarlander', extra_s)
		del learn
		gc.collect()
		# meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		# interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
		# interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		# group_interp.add_interpretation(interp)
		# filename = f'{learn.model.name.lower()}_{meta}'
		# group_interp.to_pickle(f'../docs_src/data/lunarlander_{learn.model.name.lower()}/', filename)
		# [g.write('../res/run_gifs/lunarlander') for g in interp.generate_gif()]
		# del learn


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'], list(product(p_model, p_format, p_exp)))
def test_dqn_models_mountaincar(model_cls, s_format, experience):
	group_interp = GroupAgentInterpretation()
	for i in range(5):
		learn = trained_learner(model_cls, 'MountainCar-v0', s_format, experience, bs=32, layers=[24, 12],
								memory_size=1000000, decay=0.00001, copy_over_frequency=1000)
		meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
		interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		group_interp.add_interpretation(interp)
		filename = f'{learn.model.name.lower()}_{meta}'
		group_interp.to_pickle(f'../docs_src/data/mountaincar_{learn.model.name.lower()}/', filename)
		[g.write('../res/run_gifs/mountaincar') for g in interp.generate_gif()]
		del learn
