from time import sleep

import gym
import numpy as np

from fast_rl.core.Envs import Envs
from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessDataBunch


def test_envs_all():
    msg = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    max_steps = 50
    envs = Envs.get_all_envs()

    for env in envs:
        print(f'Testing {env}')
        env_databunch = MarkovDecisionProcessDataBunch.from_env(env, max_steps=max_steps, num_workers=0)
        if env_databunch is None:
            print(f'Env {env} is probably Mujoco... Add imports if you want and try on your own. Don\'t like '
                  f'proprietary engines like this. If you have any issues, feel free to make a PR!')
            continue
        epochs = 1

        assert max_steps == len(env_databunch.train_dl)
        assert max_steps == len(env_databunch.valid_dl)

        for epoch in range(epochs):
            for element in env_databunch.train_dl:
                env_databunch.train_ds.actions = env_databunch.train_ds.get_random_action()
                # print(f'state {element} action {env_databunch.train_dl.dl.dataset.actions}')
                assert np.sum(np.equal(env_databunch.train_dl.dl.dataset.actions, env_databunch.train_ds.actions)) == \
                    np.size(env_databunch.train_ds.actions), msg

            for element in env_databunch.valid_dl:
                env_databunch.valid_ds.actions = env_databunch.valid_ds.get_random_action()
                # print(f'state {element} action {env_databunch.valid_dl.dl.dataset.actions}')
                assert np.sum(np.equal(env_databunch.train_dl.dl.dataset.actions, env_databunch.train_ds.actions)) == \
                    np.size(env_databunch.train_ds.actions), msg


def test_individualEnv():
    msg = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    max_steps = 50

    env = 'CarRacing-v0'
    print(f'Testing {env}')
    env_databunch = MarkovDecisionProcessDataBunch.from_env(env, max_steps=max_steps, num_workers=0)
    epochs = 1

    assert max_steps == len(env_databunch.train_dl)
    assert max_steps == len(env_databunch.valid_dl)

    for epoch in range(epochs):
        for element in env_databunch.train_dl:
            env_databunch.train_ds.actions = env_databunch.train_ds.get_random_action()
            # print(f'state {element.shape} action {env_databunch.train_dl.dl.dataset.actions}')
            assert np.sum(
                np.equal(env_databunch.train_dl.dl.dataset.actions, env_databunch.train_ds.actions)) == np.size(
                env_databunch.train_ds.actions), msg

        for element in env_databunch.valid_dl:
            env_databunch.valid_ds.actions = env_databunch.valid_ds.get_random_action()
            # print(f'state {element.shape} action {env_databunch.valid_dl.dl.dataset.actions}')
            assert np.sum(
                np.equal(env_databunch.train_dl.dl.dataset.actions, env_databunch.train_ds.actions)) == np.size(
                env_databunch.train_ds.actions), msg
