from functools import partial
from itertools import product

import pytest
from fastai.basic_train import torch, DatasetType
from fastai.core import ifnone

from fast_rl.agents.ddpg import create_ddpg_model, ddpg_learner, DDPGLearner
from fast_rl.agents.ddpg_models import DDPGModule
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, OrnsteinUhlenbeck
from fast_rl.core.data_block import FEED_TYPE_STATE, MDPDataBunch, ResolutionWrapper
from fast_rl.core.metrics import RewardMetric, EpsilonMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation

p_model=[DDPGModule]
p_exp=[ExperienceReplay, PriorityExperienceReplay]
p_format=[FEED_TYPE_STATE]  # , FEED_TYPE_IMAGE]
p_full_format=[FEED_TYPE_STATE]
p_envs=['Pendulum-v0']

config_env_expectations={
	'Pendulum-v0': {'action_shape': (1, 1), 'state_shape': (1, 3)}
}


def trained_learner(model_cls, env, s_format, experience, bs=64,layers=None,render='rgb_array', memory_size=1000000,
					decay=0.0001,lr=None,actor_lr=None,epochs=450,opt=torch.optim.RMSprop, **kwargs):
	lr,actor_lr=ifnone(lr,1e-3),ifnone(actor_lr,1e-4)
	data=MDPDataBunch.from_env(env,render=render,bs=bs,add_valid=False,keep_env_open=False,feed_type=s_format,
							   memory_management_strategy='k_partitions_top',k=3,**kwargs)
	exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape,epsilon_start=1,epsilon_end=0.1,
										 decay=decay)
	memory=experience(memory_size=memory_size, reduce_ram=True)
	model=create_ddpg_model(data=data,base_arch=model_cls,lr=lr,actor_lr=actor_lr,layers=layers,opt=opt)
	learner=ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
						 callback_fns=[RewardMetric, EpsilonMetric])
	learner.fit(epochs)
	return learner

def check_shape(env,data,s_format):
	assert config_env_expectations[env]['action_shape']==(1, data.action.taken_action.shape[1])
	if s_format==FEED_TYPE_STATE:
		assert config_env_expectations[env]['state_shape']==data.state.s.shape

def learner2gif(lnr:DDPGLearner,s_format,group_interp:GroupAgentInterpretation,name:str,extra:str):
	meta=f'{lnr.memory.__class__.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
	interp=AgentInterpretation(lnr, ds_type=DatasetType.Train)
	interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
	group_interp.add_interpretation(interp)
	group_interp.to_pickle(f'../docs_src/data/{name}_{lnr.model.name.lower()}/', f'{lnr.model.name.lower()}_{meta}')
	[g.write(f'../res/run_gifs/{name}_{extra}') for g in interp.generate_gif()]


@pytest.mark.parametrize(["model_cls", "s_format", "env"], list(product(p_model, p_format, p_envs)))
def test_ddpg_create_ddpg_model(model_cls, s_format, env):
	data=MDPDataBunch.from_env(env, render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
	model=create_ddpg_model(data, model_cls)
	model.eval()
	model(data.state.s.float())
	check_shape(env,data,s_format)
	data.close()


@pytest.mark.parametrize(["model_cls", "s_format", "mem", "env"], list(product(p_model, p_format, p_exp, p_envs)))
def test_ddpg_ddpglearner(model_cls, s_format, mem, env):
	data=MDPDataBunch.from_env(env, render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
	model=create_ddpg_model(data, model_cls)
	memory=mem(memory_size=1000, reduce_ram=True)
	exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
		decay=0.001)
	ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
	check_shape(env,data,s_format)
	data.close()


@pytest.mark.parametrize(["model_cls", "s_format", "mem", "env"], list(product(p_model, p_format, p_exp, p_envs)))
def test_ddpg_fit(model_cls, s_format, mem, env):
	learner=trained_learner(env=env, bs=10,opt=torch.optim.RMSprop,model_cls=model_cls,layers=[20, 20],memory_size=100,
							max_steps=20,render='rgb_array',decay=0.001,s_format=s_format,experience=mem,epochs=2)

	check_shape(env,learner.data,s_format)
	del learner


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_format, p_exp)))
def test_ddpg_models_pendulum(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	extra_s=f'{experience.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		print('\n')
		learner=trained_learner(model_cls,'Pendulum-v0',s_format,experience,decay=0.0001,render='rgb_array')
		learner2gif(learner,s_format,group_interp,'pendulum',extra_s)
		del learner


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_format, p_exp)))
def test_ddpg_models_acrobot(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	extra_s=f'{experience.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		print('\n')
		learner=trained_learner(model_cls,'Acrobot-v1',s_format,experience,decay=0.0001,render='rgb_array')
		learner2gif(learner,s_format,group_interp,'acrobot',extra_s)
		del learner


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_mountain_car_continuous(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	for i in range(5):
		print('\n')
		data=MDPDataBunch.from_env('MountainCarContinuous-v0', render='rgb_array', bs=40, add_valid=False, keep_env_open=False,
			feed_type=s_format, memory_management_strategy='k_partitions_top', k=3, res_wrap=partial(ResolutionWrapper, w_step=2, h_step=2))
		exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
			decay=0.0001)
		memory=experience(memory_size=1000000, reduce_ram=True)
		model=create_ddpg_model(data=data, base_arch=model_cls)
		learner=ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
			callback_fns=[RewardMetric, EpsilonMetric])
		learner.fit(450)

		meta=f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		interp=AgentInterpretation(learner, ds_type=DatasetType.Train)
		interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		group_interp.add_interpretation(interp)
		group_interp.to_pickle(f'../docs_src/data/mountaincarcontinuous_{model.name.lower()}/',
			f'{model.name.lower()}_{meta}')
		[g.write('../res/run_gifs/mountaincarcontinuous') for g in interp.generate_gif()]
		data.close()
		del learner
		del model
		del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_reach(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	for i in range(5):
		print('\n')
		data=MDPDataBunch.from_env('ReacherPyBulletEnv-v0', render='rgb_array', bs=40, add_valid=False, keep_env_open=False, feed_type=s_format,
			memory_management_strategy='k_partitions_top', k=3, res_wrap=partial(ResolutionWrapper, w_step=2, h_step=2))
		exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
			decay=0.00001)
		memory=experience(memory_size=1000000, reduce_ram=True)
		model=create_ddpg_model(data=data, base_arch=model_cls)
		learner=ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
			callback_fns=[RewardMetric, EpsilonMetric])
		learner.fit(450)

		meta=f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		interp=AgentInterpretation(learner, ds_type=DatasetType.Train)
		interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		group_interp.add_interpretation(interp)
		group_interp.to_pickle(f'../docs_src/data/reacher_{model.name.lower()}/',
			f'{model.name.lower()}_{meta}')
		[g.write('../res/run_gifs/reacher') for g in interp.generate_gif()]
		data.close()
		del learner
		del model
		del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_walker(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	extra_s=f'{experience.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		print('\n')
		# data=MDPDataBunch.from_env('Walker2DPyBulletEnv-v0', render='human', bs=64, add_valid=False, keep_env_open=False,
		# 	feed_type=s_format, memory_management_strategy='k_partitions_top', k=3)
		# exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
		# 	decay=0.0001)
		# memory=experience(memory_size=1000000, reduce_ram=True)
		# model=create_ddpg_model(data=data, base_arch=model_cls)
		# learner=ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
		# 	callback_fns=[RewardMetric, EpsilonMetric])
		# learner.fit(1500)
		learner=trained_learner(model_cls,'Walker2DPyBulletEnv-v0',s_format,experience,decay=0.0001,render='rgb_array')
		learner2gif(learner,s_format,group_interp,'walker2d',extra_s)

		# meta=f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		# interp=AgentInterpretation(learner, ds_type=DatasetType.Train)
		# interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		# group_interp.add_interpretation(interp)
		# group_interp.to_pickle(f'../docs_src/data/walker2d_{model.name.lower()}/',
		# 	f'{model.name.lower()}_{meta}')
		# [g.write('../res/run_gifs/walker2d') for g in interp.generate_gif()]
		# data.close()
		del learner
		# del model
		# del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_ant(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	extra_s = f'{experience.__name__}_{model_cls.__name__}_{s_format}'
	for i in range(5):
		print('\n')
		# data=MDPDataBunch.from_env('AntPyBulletEnv-v0', render='human', bs=64, add_valid=False, keep_env_open=False,
		# 	feed_type=s_format, memory_management_strategy='k_partitions_top', k=3)
		# exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
		# 	decay=0.00001)
		# memory=experience(memory_size=1000000, reduce_ram=True)
		# model=create_ddpg_model(data=data, base_arch=model_cls, lr=1e-3, actor_lr=1e-4)
		# learner=ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
		# 	opt_func=torch.optim.Adam, callback_fns=[RewardMetric, EpsilonMetric])
		# learner.fit(4)
		learner=trained_learner(model_cls,'AntPyBulletEnv-v0',s_format,experience,decay=0.0001,render='rgb_array',epochs=1000)
		learner2gif(learner,s_format,group_interp,'ant',extra_s)
		# meta=f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		# interp=AgentInterpretation(learner, ds_type=DatasetType.Train)
		# interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		# group_interp.add_interpretation(interp)
		# group_interp.to_pickle(f'../docs_src/data/ant_{model.name.lower()}/',
		# 	f'{model.name.lower()}_{meta}')
		# [g.write('../res/run_gifs/ant', frame_skip=5) for g in interp.generate_gif()]
		del learner
		# del model
		# del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
	list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_halfcheetah(model_cls, s_format, experience):
	group_interp=GroupAgentInterpretation()
	for i in range(5):
		print('\n')
		data=MDPDataBunch.from_env('HalfCheetahPyBulletEnv-v0', render='rgb_array', bs=64, add_valid=False, keep_env_open=False,
			feed_type=s_format, memory_management_strategy='k_partitions_top', k=3, res_wrap=partial(ResolutionWrapper, w_step=2, h_step=2))
		exploration_method=OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
			decay=0.000001)
		memory=experience(memory_size=1000000, reduce_ram=True)
		model=create_ddpg_model(data=data, base_arch=model_cls)
		learner=ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
			callback_fns=[RewardMetric, EpsilonMetric])
		learner.fit(1000)

		meta=f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format==FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
		interp=AgentInterpretation(learner, ds_type=DatasetType.Train)
		interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
		group_interp.add_interpretation(interp)
		group_interp.to_pickle(f'../docs_src/data/halfcheetah_{model.name.lower()}/',
			f'{model.name.lower()}_{meta}')
		[g.write('../res/run_gifs/halfcheetah') for g in interp.generate_gif()]
		del learner
		del model
		del data
