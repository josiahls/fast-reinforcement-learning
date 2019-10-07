import gym
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner, ItemLists
from fastai.vision import ImageDataBunch
import numpy as np

from fast_rl.agents.DQN import FixedTargetDQN
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice, MarkovDecisionProcessList, \
    MDPDataBunch, MDPDataset
from fast_rl.core.agent_core import ExperienceReplay


def test_MarkovDecisionProcessDataBunch_init():
    error_msg = 'state space is %s but should be %s'
    error_msg2 = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    max_steps = 50
    # Create 2 itemlists
    train_list = MDPDataset(gym.make('CartPole-v1'), max_steps=max_steps)
    valid_list = MDPDataset(gym.make('CartPole-v1'), max_steps=max_steps)

    env_databunch = MDPDataBunch.create(train_list, valid_list, num_workers=0)
    epochs = 1

    assert max_steps == len(train_list)
    assert max_steps == len(train_list)
    assert max_steps == len(env_databunch.train_dl)
    assert max_steps == len(env_databunch.valid_dl)

    for epoch in range(epochs):
        for element in env_databunch.train_dl:
            env_databunch.train_ds.actions = env_databunch.train_ds.env.action_space.sample()
            current_s, actual_s = element.shape[1], train_list.env.observation_space.shape[0]
            print(f'state {element} action {env_databunch.train_dl.dl.dataset.actions}')
            assert current_s == actual_s, error_msg % (current_s, actual_s)
            assert np.equal(env_databunch.train_dl.dl.dataset.actions, env_databunch.train_ds.actions), error_msg2

        for element in env_databunch.valid_dl:
            env_databunch.valid_ds.actions = env_databunch.valid_ds.env.action_space.sample()
            current_s, actual_s = element.shape[1], valid_list.env.observation_space.shape[0]
            print(f'state {element} action {env_databunch.valid_dl.dl.dataset.actions}')
            assert current_s == actual_s, error_msg % (current_s, actual_s)
            assert np.equal(env_databunch.valid_dl.dl.dataset.actions, env_databunch.valid_ds.actions), error_msg2


def test_MDPDataset_MemoryManagement():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, batch_size=128, max_episodes=10000, lr=0.001, copy_over_frequency=3,
                           memory=ExperienceReplay(10000), discount=0.99)
    learn = AgentLearner(data, model, mem_strategy='k_top_best')

    learn.fit(5)


def test_MarkovDecisionProcessDataBunch_init_no_valid():
    error_msg = 'state space is %s but should be %s'
    error_msg2 = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    max_steps = 50
    # Create 2 itemlists
    train_list = MDPDataset(gym.make('CartPole-v1'), max_steps=max_steps)

    env_databunch = MDPDataBunch.create(train_list, num_workers=0)
    env_databunch.valid_dl = None
    epochs = 3

    assert max_steps == len(train_list)
    assert max_steps == len(train_list)
    assert max_steps == len(env_databunch.train_dl)
    assert env_databunch.valid_dl is None

    for epoch in range(epochs):
        print(f'epoch {epoch}')
        for element in env_databunch.train_dl:
            env_databunch.train_ds.actions = env_databunch.train_ds.env.action_space.sample()
            current_s, actual_s = element.shape[1], train_list.env.observation_space.shape[0]
            print(f'state {element} action {env_databunch.train_dl.dl.dataset.actions}')
            assert current_s == actual_s, error_msg % (current_s, actual_s)
            assert np.equal(env_databunch.train_dl.dl.dataset.actions, env_databunch.train_ds.actions), error_msg2
