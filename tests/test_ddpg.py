from functools import partial
from itertools import product

import pytest
from fast_rl.agents.ddpg_models import DDPGModule
from fastai.basic_train import torch, DatasetType

from fast_rl.agents.ddpg import create_ddpg_model, ddpg_learner
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, OrnsteinUhlenbeck
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import FEED_TYPE_STATE, MDPDataBunch, FEED_TYPE_IMAGE
from fast_rl.core.metrics import RewardMetric, EpsilonMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation


p_model = [DDPGModule]
p_exp = [ExperienceReplay, PriorityExperienceReplay]
p_format = [FEED_TYPE_STATE]#, FEED_TYPE_IMAGE]
p_full_format = [FEED_TYPE_STATE]
p_envs = ['Walker2DPyBulletEnv-v0']

config_env_expectations = {
    'Walker2DPyBulletEnv-v0': {'action_shape': (1, 6), 'state_shape': (1, 22)}
}


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "env"], list(product(p_model, p_format, p_envs)))
def test_ddpg_create_ddpg_model(model_cls, s_format, env):
    data = MDPDataBunch.from_env(env, render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
    model = create_ddpg_model(data, model_cls)
    model.eval()
    model(data.state.s.float())

    assert config_env_expectations[env]['action_shape'] == (1, data.action.taken_action.shape[1])
    if s_format == FEED_TYPE_STATE:
        assert config_env_expectations[env]['state_shape'] == data.state.s.shape


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "mem", "env"], list(product(p_model, p_format, p_exp, p_envs)))
def test_dddpg_ddpglearner(model_cls, s_format, mem, env):
    data = MDPDataBunch.from_env(env, render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
    model = create_ddpg_model(data, model_cls)
    memory = mem(memory_size=1000, reduce_ram=True)
    exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1, decay=0.001)
    ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)

    assert config_env_expectations[env]['action_shape'] == (1, data.action.taken_action.shape[1])
    if s_format == FEED_TYPE_STATE:
        assert config_env_expectations[env]['state_shape'] == data.state.s.shape


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", "mem", "env"], list(product(p_model, p_format, p_exp, p_envs)))
def test_ddpg_fit(model_cls, s_format, mem, env):
    data = MDPDataBunch.from_env(env, render='rgb_array', bs=10, max_steps=20, add_valid=False, feed_type=s_format)
    model = create_ddpg_model(data, model_cls, opt=torch.optim.RMSprop, layers=[20, 20])
    memory = mem(memory_size=100, reduce_ram=True)
    exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1, decay=0.001)
    learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                           callback_fns=[RewardMetric, EpsilonMetric])
    learner.fit(2)

    assert config_env_expectations[env]['action_shape'] == (1, data.action.taken_action.shape[1])
    if s_format == FEED_TYPE_STATE:
        assert config_env_expectations[env]['state_shape'] == data.state.s.shape

    del data
    del model
    del learner


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(p_model, p_format, p_exp)))
def test_ddpg_models_pendulum(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        print('\n')
        data = MDPDataBunch.from_env('Pendulum-v0', render='human', bs=64, add_valid=False, feed_type=s_format)
        exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
                                               decay=0.0001)
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = create_ddpg_model(data=data, base_arch=model_cls)
        learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                               callback_fns=[RewardMetric, EpsilonMetric])
        learner.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learner, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/pendulum_{model.name.lower()}/', f'{model.name.lower()}_{meta}')

        del learner
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_mountain_car_continuous(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        print('\n')
        data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human', bs=40, add_valid=False, feed_type=s_format)
        exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
                                               decay=0.0001)
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = create_ddpg_model(data=data, base_arch=model_cls)
        learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                               callback_fns=[RewardMetric, EpsilonMetric])
        learner.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learner, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/mountaincarcontinuous_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')

        del learner
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_reach(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        print('\n')
        data = MDPDataBunch.from_env('ReacherPyBulletEnv-v0', render='human', bs=40, add_valid=False,feed_type=s_format)
        exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
                                               decay=0.0001)
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = create_ddpg_model(data=data, base_arch=model_cls)
        learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                               callback_fns=[RewardMetric, EpsilonMetric])
        learner.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learner, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/reacher_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')

        del learner
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_walker(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        print('\n')
        data = MDPDataBunch.from_env('Walker2DPyBulletEnv-v0', render='human', bs=64, add_valid=False,
                                     feed_type=s_format)
        exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
                                               decay=0.0001)
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = create_ddpg_model(data=data, base_arch=model_cls)
        learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                               callback_fns=[RewardMetric, EpsilonMetric])
        learner.fit(2000)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learner, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/walker2d_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')

        del learner
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_ant(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        print('\n')
        data = MDPDataBunch.from_env('AntPyBulletEnv-v0', render='human', bs=64, add_valid=False,feed_type=s_format)
        exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
                                               decay=0.00001)
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = create_ddpg_model(data=data, base_arch=model_cls, lr=1e-3, actor_lr=1e-4,)
        learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                               opt_func=torch.optim.Adam, callback_fns=[RewardMetric, EpsilonMetric])
        learner.fit(1000)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learner, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/ant_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')

        del learner
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(p_model, p_full_format, p_exp)))
def test_ddpg_models_halfcheetah(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        print('\n')
        data = MDPDataBunch.from_env('HalfCheetahPyBulletEnv-v0', render='human', bs=64, add_valid=False,
                                     feed_type=s_format)
        exploration_method = OrnsteinUhlenbeck(size=data.action.taken_action.shape, epsilon_start=1, epsilon_end=0.1,
                                               decay=0.0001)
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = create_ddpg_model(data=data, base_arch=model_cls)
        learner = ddpg_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                               callback_fns=[RewardMetric, EpsilonMetric])
        learner.fit(2000)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learner, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/halfcheetah_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')

        del learner
        del model
        del data