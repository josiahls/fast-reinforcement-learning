import gym
import pytest
import torch
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner, ItemLists, fit
from fastai.vision import ImageDataBunch
import numpy as np
from gym import error
from gym.envs.algorithmic.algorithmic_env import AlgorithmicEnv
from gym.envs.toy_text import discrete
from gym.wrappers import TimeLimit

from fast_rl.agents.DQN import FixedTargetDQN, DQN
from fast_rl.core.Envs import Envs
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSliceAlpha, MarkovDecisionProcessListAlpha, \
    MDPDataBunchAlpha, MDPDatasetAlpha, Action, Bounds, State, MDPDataset, MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.util.exceptions import MaxEpisodeStepsMissingError
from fast_rl.util.misc import list_in_str

ENV_NAMES = Envs.get_all_latest_envs()


def validate_item_list(item_list: ItemLists):
    # Check items
    for i, item in enumerate(item_list.items):
        if item.done: assert not item_list.items[
            i - 1].done, f'The dataset has duplicate "done\'s" that are consecutive.'
        assert item.state.s is not None, f'The item: {item}\'s state is None'
        assert item.state.s_prime is not None, f'The item: {item}\'s state prime is None'


@pytest.mark.parametrize("env", sorted(['CartPole-v0']))
def test_mdp_callback(env):
    data = MDPDataBunch.from_env(env, render='rgb_array')
    model = DQN(data)
    learner = AgentLearner(data, model)
    fit(5, learner, learner.callbacks, learner.metrics)



@pytest.mark.parametrize("env", sorted(['CartPole-v0']))
def test_mdp_databunch(env):
    data = MDPDataBunch.from_env(env, add_valid=False, render='rgb_array')
    for i in range(5):
        for _ in data.train_ds:
            data.train_ds.action = Action(taken_action=data.train_ds.action.action_space.sample(),
                                          action_space=data.train_ds.action.action_space)

    validate_item_list(data.train_ds.x)


@pytest.mark.parametrize("env", sorted(['CartPole-v0']))
def test_mdp_dataset_iter(env):
    dataset = MDPDataset(gym.make(env), memory_manager=None, bs=8, render='rgb_array')

    for epoch in range(5):
        for el in dataset:
            dataset.action.set_single_action(dataset.env.action_space.sample())

    # Check items
    for i, item in enumerate(dataset.x.items):
        if item.done: assert not dataset.x.items[
            i - 1].done, f'The dataset has duplicate "done\'s" that are consecutive.'
        assert item.state.s is not None, f'The item: {item}\'s state is None'
        assert item.state.s_prime is not None, f'The item: {item}\'s state prime is None'


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

    action = Action(taken_action=taken_action, raw_action=raw_action, action_space=init_env.action_space)

    if list_in_str(env, ['mountaincar-', 'cartpole', 'pong']):
        assert any([action.taken_action.dtype in (int, torch.int, torch.int64)]), f'Action is wrong dtype {action}'
        assert any([action.raw_action.dtype in (float, torch.float32, torch.float64)]), f'Action is wrong dtype {action}'
    if list_in_str(env, ['carracing', 'pendulum']):
        assert any([action.taken_action.dtype in (float, torch.float32, torch.float64)]), f'Action is wrong dtype {action}'
        assert any([action.raw_action.dtype in (float, torch.float32, torch.float64)]), f'Action is wrong dtype {action}'


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


@pytest.mark.parametrize("env", sorted(ENV_NAMES))
def test_state_str(env):
    try:
        init_env = gym.make(env)
    except error.DependencyNotInstalled as e:
        print(e)
        return

    render = 'rgb_array'
    if isinstance(init_env, TimeLimit) and isinstance(init_env.unwrapped, (AlgorithmicEnv, discrete.DiscreteEnv)):
        render = 'ansi' if render == 'rgb_array' else render

    taken_action = init_env.action_space.sample()
    state = init_env.reset()

    try:
        alt_s = init_env.render(render)
    except NotImplementedError:
        return

    state_prime, reward, done, info = init_env.step(taken_action)
    alt_s_prime = init_env.render(render)
    State(state, state_prime, alt_s, alt_s_prime, init_env.observation_space).__str__()
    init_env.close()


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
    init_env.close()