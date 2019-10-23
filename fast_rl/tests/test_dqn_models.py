from functools import partial
from itertools import  product
from time import sleep

import pygame
import pytest

from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch, FEED_TYPE_STATE, FEED_TYPE_IMAGE
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.core.basic_train import AgentLearner


params_dqn = [DuelingDQN, DoubleDQN, DQN, FixedTargetDQN, DoubleDuelingDQN]
params_envs = ['Pong-v0']#, 'CartPole-v0', 'maze-random-5x5-v0']
params_state_format = [FEED_TYPE_STATE, FEED_TYPE_IMAGE]


@pytest.mark.parametrize(["env", "model", "s_format"], list(product(params_envs, params_dqn, params_state_format)))
def test_dqn_models(env, model, s_format):
    model = partial(model, memory=ExperienceReplay(memory_size=100000, reduce_ram=True))
    print('\n')
    maze_errored, counter, data = True, 0, None
    while maze_errored:
        try:
            data = MDPDataBunch.from_env(env, render='rgb_array', max_steps=100, add_valid=False, feed_type=s_format)
            maze_errored = False
        except pygame.error as e:
            counter += 1
            sleep(1)
            if counter > 5: assert False, f'Failed to many times, got {e}'

    learn = AgentLearner(data, model(data))
    learn.fit(5)
    del learn
    del model
    del data
