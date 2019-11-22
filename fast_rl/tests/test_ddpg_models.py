from functools import partial
from itertools import product

import pytest
from fastai.basic_train import torch, DatasetType

from fast_rl.agents.ddpg import DDPG
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, OrnsteinUhlenbeck
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import FEED_TYPE_STATE, MDPDataBunch
from fast_rl.core.metrics import RewardMetric, EpsilonMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation

params_ddpg = [DDPG]
params_experience = [ExperienceReplay, PriorityExperienceReplay]
params_state_format = [FEED_TYPE_STATE]  # , FEED_TYPE_IMAGE]


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(params_ddpg, params_state_format, params_experience)))
def test_ddpg_models_pendulum(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        memory = experience(memory_size=1000000, reduce_ram=True)
        model = partial(model_cls, memory=memory, opt=torch.optim.RMSprop)

        print('\n')
        data = MDPDataBunch.from_env('Pendulum-v0', render='human', bs=64, add_valid=False,
                                     feed_type=s_format)
        model = model(data)
        learn = AgentLearner(data, model, callback_fns=[RewardMetric, EpsilonMetric])
        learn.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../../docs_src/data/pendulum_{model.name.lower()}/', f'{model.name.lower()}_{meta}')

        del learn
        del model
        del data


@pytest.mark.usefixtures('skip_performance_check')
@pytest.mark.parametrize(["model_cls", "s_format", 'experience'],
                         list(product(params_ddpg, params_state_format, params_experience)))
def test_ddpg_models_mountain_car_continuous(model_cls, s_format, experience):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        memory = experience(memory_size=1000000, reduce_ram=True)

        print('\n')
        data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human', bs=64, add_valid=False,
                                     feed_type=s_format)

        model = partial(model_cls, memory=memory, opt=torch.optim.Adam,
                        exploration_strategy=OrnsteinUhlenbeck(size=data.action.taken_action.shape,
                                                               epsilon_start=1, epsilon_end=0.1,
                                                               decay=0.00001,
                                                               do_exploration=True))
        model = model(data)
        learn = AgentLearner(data, model, callback_fns=[RewardMetric, EpsilonMetric])
        learn.fit(450)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../../docs_src/data/mountaincarcontinuous_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')

        del learn
        del model
        del data
