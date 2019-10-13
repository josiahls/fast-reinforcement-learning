from collections import Collection

import pytest
from fastai.basic_train import LearnerCallback

from fast_rl.agents.DDPG import DDPG
from fast_rl.core.Envs import Envs
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay, OrnsteinUhlenbeck


ENV_NAMES = Envs.get_all_latest_envs('pybullet')


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_all_ddpg(env):
    data = MDPDataBunch.from_env(env, render='rgb_array', add_valid=False,
                                 max_steps=50)
    if data is None:
        print(f'Env {env} is probably Mujoco... Add imports if you want and try on your own. Don\'t like '
              f'proprietary engines like this. If you have any issues, feel free to make a PR!')
        return
    # If the action type is int, skip. While there is a capability of int actions spaces for DDPG,
    # that functionality is more of a niche
    if data.train_ds.env.action_space.dtype == int: return
    # If the dtype is None, then it is most likely a Tuple action space. We will most likely open this
    # functionality in the future.
    if data.train_ds.env.action_space.dtype is None: return

    # data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human')
    model = DDPG(data=data, batch=8, memory=ExperienceReplay(200, reduce_ram=True),
                 exploration_strategy=OrnsteinUhlenbeck(epsilon_start=1, epsilon_end=0.1, decay=0.0001, size=1,
                                                        do_exploration=True, end_episode=450))
    learn = AgentLearner(data, model)
    learn.fit(5)
