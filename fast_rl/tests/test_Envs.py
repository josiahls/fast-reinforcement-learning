import gym
import numpy as np
import pytest
from fast_rl.core.Envs import Envs
from fast_rl.core.MarkovDecisionProcess import MDPDataBunchAlpha




def test_individual_env():
    msg = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    max_steps = 50

    env = 'CarRacing-v0'
    print(f'Testing {env}')
    mdp_databunch = MDPDataBunchAlpha.from_env(env, max_steps=max_steps, num_workers=0)
    epochs = 1

    assert max_steps == len(mdp_databunch.train_dl)
    assert max_steps == len(mdp_databunch.valid_dl)

    for epoch in range(epochs):
        for _ in mdp_databunch.train_dl:
            mdp_databunch.train_ds.actions = mdp_databunch.train_ds.get_random_action()
            # print(f'state {element.shape} action {mdp_databunch.train_dl.dl.dataset.actions}')
            assert np.sum(
                np.equal(mdp_databunch.train_dl.dl.dataset.actions, mdp_databunch.train_ds.actions)) == np.size(
                mdp_databunch.train_ds.actions), msg

        for _ in mdp_databunch.valid_dl:
            mdp_databunch.valid_ds.actions = mdp_databunch.valid_ds.get_random_action()
            # print(f'state {element.shape} action {mdp_databunch.valid_dl.dl.dataset.actions}')
            assert np.sum(
                np.equal(mdp_databunch.train_dl.dl.dataset.actions, mdp_databunch.train_ds.actions)) == np.size(
                mdp_databunch.train_ds.actions), msg


def test_individual_env_no_dl():
    """Just a nice place to do sanity testing on new / untested envs."""
    env = gym.make('maze-random-10x10-plus-v0')
    for episode in range(2):
        done = False
        env.reset()
        while not done:
            output = env.step(env.action_space.sample())
            done = output[2]
            env.render('human')
