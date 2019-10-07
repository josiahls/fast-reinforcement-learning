from fastai.basic_train import LearnerCallback, DatasetType
from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch, FEED_TYPE_STATE
from fast_rl.core.agent_core import PriorityExperienceReplay, ExperienceReplay


def test_priority_experience_replay():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = FixedTargetDQN(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(5)


def test_stripped_fit():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 5

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, n_epochs=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(epoch=epoch) for c in callbacks]
        learn.model.train()
        for element in learn.data.train_dl:
            learn.data.train_ds.actions = learn.predict(element)
            [c.on_step_end(learn=learn) for c in callbacks]
        [c.on_epoch_end() for c in callbacks]

        # For now we are going to avoid executing learner_callbacks here.
        learn.model.eval()
        for element in learn.data.valid_dl:
            learn.data.valid_ds.actions = learn.predict(element)

        if epoch % 1 == 0:
            interp = AgentInterpretationAlpha(learn)
            interp.plot_heatmapped_episode(epoch)
    [c.on_train_end() for c in callbacks]


def test_fit_function_ddpg():
    data = MDPDataBunch.from_env('Pendulum-v0', render='human', max_steps=1000)
    model = DDPG(data, memory=PriorityExperienceReplay(1000))
    learn = AgentLearner(data, model)
    learn.fit(5)


def test_fit_function_dqn():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000, add_valid=False,
                                 feed_type=FEED_TYPE_STATE)
    model = FixedTargetDQN(data, batch_size=128, max_episodes=10000, lr=0.001, copy_over_frequency=3,
                           memory=ExperienceReplay(10000), discount=0.99)
    learn = AgentLearner(data, model)

    learn.fit(5)
    interp = AgentInterpretationAlpha(learn, base_chart_size=(20, 10))
    interp.plot_heatmapped_episode(-1, return_heat_maps=False)

