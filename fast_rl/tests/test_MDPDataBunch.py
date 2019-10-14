import gym
import pytest
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner, ItemLists
from fastai.vision import ImageDataBunch
import numpy as np
from gym import error

from fast_rl.agents.DQN import FixedTargetDQN
from fast_rl.core.Envs import Envs
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSliceAlpha, MarkovDecisionProcessListAlpha, \
    MDPDataBunchAlpha, MDPDatasetAlpha, Action, Bounds, State
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.util.misc import list_in_str

ENV_NAMES = Envs.get_all_latest_envs()


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_bound_data_structure(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    for bound in (Bounds(init_env.action_space), Bounds(init_env.observation_space)):
        if env.lower().__contains__('continuous'):
            assert bound.n_possible_values == np.inf, f'Env {env} is continuous, should have inf values.'
        if env.lower().__contains__('deterministic'):
            assert bound.n_possible_values != np.inf, f'Env {env} is deterministic, should have discrete values.'


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_action_data_structure(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    taken_action = init_env.action_space.sample()
    raw_action = np.random.rand(len(Bounds(init_env.action_space).max))
    init_env.reset()
    _ = init_env.step(taken_action)

    action = Action(taken_action, raw_action, init_env.action_space)

    if list_in_str(env, ['mountaincar-', 'cartpole', 'pong']):
        assert any([action.taken_action.dtype in (int, np.int, np.int64)]), f'Action is wrong dtype {action}'
        assert any([action.raw_action.dtype in (float, np.float)]), f'Action is wrong dtype {action}'
    if list_in_str(env, ['carracing', 'pendulum']):
        assert any([action.taken_action.dtype in (float, np.float, np.float32)]), f'Action is wrong dtype {action}'
        assert any([action.raw_action.dtype in (float, np.float, np.float32)]), f'Action is wrong dtype {action}'


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_state_data_structure(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    taken_action = init_env.action_space.sample()
    state = init_env.reset()
    state_prime, reward, done, info = init_env.step(taken_action)

    State(state, state_prime, init_env.observation_space)
