from functools import partial
from itertools import  product

import pytest

from fast_rl.agents.dqn import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_STATE, FEED_TYPE_IMAGE
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.core.basic_train import AgentLearner


params_dqn = [DuelingDQN, DoubleDQN, DQN, FixedTargetDQN, DoubleDuelingDQN]
params_envs = ['CartPole-v0', 'MountainCar-v0', 'Pong-v0']
params_state_format = [FEED_TYPE_STATE, FEED_TYPE_IMAGE]


@pytest.mark.parametrize(["env", "model", "s_format"], list(product(params_envs, params_dqn, params_state_format)))
def test_dqn_models(env, model, s_format):
    model = partial(model, memory=ExperienceReplay(memory_size=1000, reduce_ram=True))
    print('\n')
    data = MDPDataBunch.from_env(env, render='rgb_array', max_steps=20, bs=4, add_valid=False, feed_type=s_format)
    learn = AgentLearner(data, model(data))
    # if s_format == FEED_TYPE_STATE:
    learn.fit(3)
    # else: learn.fit(1)

    data.train_ds.env.close()

    del learn
    del model
    del data
