from functools import partial
from itertools import  product

import pytest

from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch, FEED_TYPE_STATE, FEED_TYPE_IMAGE
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.core.basic_train import AgentLearner


params_dqn = [DuelingDQN, DoubleDQN, DQN, FixedTargetDQN, DoubleDuelingDQN]
params_envs = ['CartPole-v0', 'MountainCar-v0', 'Pong-v0']
params_state_format = [FEED_TYPE_STATE, FEED_TYPE_IMAGE]


@pytest.mark.parametrize(["env", "model", "s_format"], list(product(params_envs, params_dqn, params_state_format)))
def test_dqn_models(env, model, s_format):
    model = partial(model, memory=ExperienceReplay(memory_size=1000, reduce_ram=True))
    print('\n')

    data = MDPDataBunch.from_env(env, render='rgb_array', max_steps=20, bs=4, add_valid=False,
                                 feed_type=s_format)

    learn = AgentLearner(data, model(data))

    data.train_ds.env.close()

    learn.fit(3)
    del learn
    del model
    del data
