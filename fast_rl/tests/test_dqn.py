from functools import partial
from itertools import  product

import pytest
import torch
from fastai.basic_data import DatasetType

from fast_rl.agents.dqn import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_STATE, FEED_TYPE_IMAGE
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, GreedyEpsilon
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.metrics import RewardMetric, EpsilonMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation


params_dqn = [DuelingDQN, DoubleDQN, DQN, FixedTargetDQN, DoubleDuelingDQN]
params_experience = [PriorityExperienceReplay, ExperienceReplay]
params_state_format = [FEED_TYPE_STATE]#, FEED_TYPE_IMAGE]


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(params_dqn, params_state_format, params_experience)))
def test_dqn_models_cartpole(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = partial(model_cls, memory=memory, layers=[64, 64], discount=0.99, grad_clip=1,
                        optimizer=torch.optim.RMSprop)

        if isinstance(model, DQN):
            model = partial(model, lr=0.001)
        else:
            model = partial(model, lr=0.00025, copy_over_frequency=300)

        print('\n')
        data = MDPDataBunch.from_env('CartPole-v1', render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
        model = model(data)
        learn = AgentLearner(data, model, callback_fns=[RewardMetric, EpsilonMetric])
        learn.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/cartpole_{model.name.lower()}/', f'{model.name.lower()}_{meta}')

        del learn
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(params_dqn, params_state_format, params_experience)))
def test_dqn_models_lunarlander(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = partial(model_cls, memory=memory, layers=[128, 64], discount=0.99, grad_clip=1,
                        optimizer=torch.optim.RMSprop)

        if isinstance(model, DQN):
            model = partial(model, lr=0.001)
        else:
            model = partial(model, lr=0.00025, copy_over_frequency=600)

        print('\n')
        data = MDPDataBunch.from_env('LunarLander-v2', render='rgb_array', bs=32, add_valid=False, feed_type=s_format)
        model = model(data)
        learn = AgentLearner(data, model, callback_fns=[RewardMetric, EpsilonMetric])
        learn.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/cartpole_{model.name.lower()}/', f'{model.name.lower()}_{meta}')

        del learn
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(params_dqn, params_state_format, params_experience)))
def test_dqn_models_mountaincar(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = partial(model_cls, memory=memory, layers=[24, 12], discount=0.99, grad_clip=5,
                        optimizer=torch.optim.RMSprop,
                        exploration_strategy=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.0001))

        if isinstance(model, DQN):
            model = partial(model, lr=0.001)
        else:
            model = partial(model, lr=0.001, copy_over_frequency=1000)

        print('\n')
        data = MDPDataBunch.from_env('MountainCar-v0', render='rgb_array', bs=128, add_valid=False, feed_type=s_format)
        model = model(data)
        learn = AgentLearner(data, model, callback_fns=[RewardMetric, EpsilonMetric])
        learn.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../docs_src/data/mountaincar_{model.name.lower()}/', f'{model.name.lower()}_{meta}')

        del learn
        del model
        del data