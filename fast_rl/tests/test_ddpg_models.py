from collections import Collection
from functools import partial
from itertools import product

import pytest
from fastai.basic_train import LearnerCallback

from fast_rl.agents.ddpg import DDPG
from fast_rl.core.Envs import Envs
from fast_rl.core.data_block import FEED_TYPE_IMAGE, FEED_TYPE_STATE, MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay, OrnsteinUhlenbeck
from fast_rl.core.basic_train import AgentLearner

params_dqn = [DDPG]
params_envs = ['Pendulum-v0', 'CarRacing-v0']
params_state_format = [FEED_TYPE_STATE, FEED_TYPE_IMAGE]


@pytest.mark.parametrize(["env", "model", "s_format"], list(product(params_envs, params_dqn, params_state_format)))
def test_ddpg_models(env, model, s_format):
    model = partial(model, memory=ExperienceReplay(memory_size=1000, reduce_ram=True))
    data = MDPDataBunch.from_env(env, render='rgb_array', max_steps=20, bs=4, add_valid=False, feed_type=s_format)
    learn = AgentLearner(data, model(data))
    learn.fit(3)
    data.train_ds.env.close()
    del learn
    del model
    del data
