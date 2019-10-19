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
    MDPDataBunchAlpha, MDPDatasetAlpha, Action, Bounds, State, MDPDataset
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.util.exceptions import MaxEpisodeStepsMissingError
from fast_rl.util.misc import list_in_str

ENV_NAMES = Envs.get_all_latest_envs()


@pytest.mark.parametrize("env", sorted(['CartPole-v0']))
def test_mdp_dataset_iter(env):
    dataset = MDPDataset(gym.make(env), memory_manager=None, bs=8, render='rgb_array')

    for epoch in range(5):
        for el in dataset:
            dataset.action.set_single_action(dataset.env.action_space.sample())

    print('hello')


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_mdpdataset_init(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    data = MDPDataset(init_env, None, 64, 'rgb_array')

    try:
        max_steps = data.max_steps
        assert max_steps is not None, f'Max steps is None for env {env}'
    except MaxEpisodeStepsMissingError as e:
        return

    envs_to_test = {
        'CartPole-v0': 200,
        'MountainCar-v0': 200,
        'maze-v0': 2000
    }

    if env in envs_to_test:
        assert envs_to_test[env] == max_steps, f'Env {env} is the wrong step amount'


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_bound_init(env):
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
def test_action_init(env):
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
def test_state_init(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    taken_action = init_env.action_space.sample()
    state = init_env.reset()
    state_prime, reward, done, info = init_env.step(taken_action)
    State(state, state_prime, init_env.observation_space)


@pytest.mark.parametrize("env", sorted(['CartPole-v0', 'maze-random-5x5-v0']))
def test_state_full_episode(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    done = False
    state = init_env.reset()
    while not done:
        taken_action = init_env.action_space.sample()
        alt_state = init_env.render('rgb_array')
        state_prime, reward, done, info = init_env.step(taken_action)
        alt_s_prime = init_env.render('rgb_array')
        State(state, state_prime, alt_state, alt_s_prime, init_env.observation_space)
        state = state_prime
        if done:
            assert state_prime is not None, 'State prime is None, this should not have happened.'



