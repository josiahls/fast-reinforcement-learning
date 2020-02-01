import os
from functools import partial
from itertools import product

import gym
import pytest
import numpy as np
import torch
from fastai.basic_train import ItemLists

from fast_rl.agents.dqn import create_dqn_model, dqn_learner
from fast_rl.agents.dqn_models import DQNModule
from fast_rl.core.agent_core import GreedyEpsilon, ExperienceReplay
from fast_rl.core.data_block import MDPDataBunch, ResolutionWrapper, FEED_TYPE_IMAGE
from fast_rl.core.metrics import RewardMetric, EpsilonMetric


def validate_item_list(item_list: ItemLists):
    # Check items
    for i, item in enumerate(item_list.items):
        if item.done: assert not item_list.items[
            i - 1].done, f'The dataset has duplicate "done\'s" that are consecutive.'
        assert item.state.s is not None, f'The item: {item}\'s state is None'
        assert item.state.s_prime is not None, f'The item: {item}\'s state prime is None'


@pytest.mark.parametrize(["memory_strategy", "k"], list(product(['k_top', 'k_partitions_top'], [1, 3, 5])))
def test_dataset_memory_manager(memory_strategy, k):
    data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20, add_valid=False,
                                 memory_management_strategy=memory_strategy, k=k)
    model = create_dqn_model(data, DQNModule, opt=torch.optim.RMSprop, lr=0.1)
    memory = ExperienceReplay(memory_size=1000, reduce_ram=True)
    exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
    learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                          callback_fns=[RewardMetric, EpsilonMetric])
    learner.fit(10)

    data_info = {episode: data.train_ds.x.info[episode] for episode in data.train_ds.x.info if episode != -1}
    full_episodes = [episode for episode in data_info if not data_info[episode][1]]

    assert sum([not _[1] for _ in data_info.values()]) == k, 'There should be k episodes but there is not.'
    if memory_strategy.__contains__('top') and not memory_strategy.__contains__('both'):
        assert (np.argmax([_[0] for _ in data_info.values()])) in full_episodes


def test_databunch_to_pickle():
    data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=20, add_valid=False,
                                 memory_management_strategy='k_partitions_top', k=3)
    model = create_dqn_model(data, DQNModule, opt=torch.optim.RMSprop, lr=0.1)
    memory = ExperienceReplay(memory_size=1000, reduce_ram=True)
    exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
    learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                          callback_fns=[RewardMetric, EpsilonMetric])
    learner.fit(10)
    data.to_pickle('./data/cartpole_10_epoch')
    MDPDataBunch.from_pickle(env_name='CartPole-v0', path='./data/cartpole_10_epoch')


def test_resolution_wrapper():
    data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=5, max_steps=10, add_valid=False,
                                memory_management_strategy='k_top', k=1, feed_type=FEED_TYPE_IMAGE,
                                res_wrap=partial(ResolutionWrapper, w_step=2, h_step=2))
    model = create_dqn_model(data, DQNModule, opt=torch.optim.RMSprop, lr=0.1,channels=[32,32,32],ks=[5,5,5],stride=[2,2,2])
    memory = ExperienceReplay(memory_size=1000, reduce_ram=True)
    exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
    learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method,
                          callback_fns=[RewardMetric, EpsilonMetric])
    learner.fit(2)
    temp = gym.make('CartPole-v0')
    temp.reset()
    original_shape = temp.render(mode='rgb_array').shape
    assert data.env.render(mode='rgb_array').shape == (original_shape[0] // 2, original_shape[1] // 2, 3)
