import gym
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner, ItemLists
from fastai.vision import ImageDataBunch

from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice, MarkovDecisionProcessList, \
    MarkovDecisionProcessDataBunch, MarkovDecisionProcessDataset


def test_MarkovDecisionProcessSlice_init():
    env = gym.make('CartPole-v0')

    # Create 2 itemlists
    train_list = MarkovDecisionProcessList()
    valid_list = MarkovDecisionProcessList()

    train_list = MarkovDecisionProcessDataset(gym.make('CartPole-v0'))
    valid_list = MarkovDecisionProcessDataset(gym.make('CartPole-v0'))

    env_databunch = MarkovDecisionProcessDataBunch.create(train_list, valid_list, num_workers=0)

    for element in env_databunch.train_dl:
        print(element)
