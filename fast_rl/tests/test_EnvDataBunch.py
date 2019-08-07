import gym
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner, ItemLists
from fastai.vision import ImageDataBunch
import numpy as np

from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice, MarkovDecisionProcessList, \
    MarkovDecisionProcessDataBunch, MarkovDecisionProcessDataset


def test_MarkovDecisionProcessSlice_init():
    error_msg = 'state space is %s but should be %s'
    error_msg2 = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    max_steps = 50
    # Create 2 itemlists
    train_list = MarkovDecisionProcessDataset(gym.make('CartPole-v1'), max_steps=max_steps)
    valid_list = MarkovDecisionProcessDataset(gym.make('CartPole-v1'), max_steps=max_steps)

    env_databunch = MarkovDecisionProcessDataBunch.create(train_list, valid_list, num_workers=0)
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
