from fastai.basic_data import DatasetType
from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import DQN
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


def test_interpretation_heatmap():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 10

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, max_episodes=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(episode=epoch) for c in callbacks]
        learn.model.train()
        for element in learn.data.train_dl:
            learn.data.train_ds.actions = learn.predict(element)
            [c.on_step_end(learn=learn) for c in callbacks]
        [c.on_epoch_end() for c in callbacks]

        # For now we are going to avoid executing callbacks here.
        learn.model.eval()
        for element in learn.data.valid_dl:
            learn.data.valid_ds.actions = learn.predict(element)

        if epoch % 1 == 0:
            interp = AgentInterpretationAlpha(learn)
            interp.plot_heatmapped_episode(epoch)
    [c.on_train_end() for c in callbacks]


def test_interpretation_plot_sequence():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 20

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, max_episodes=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(episode=epoch) for c in callbacks]
        learn.model.train()
        counter = 0
        for element in learn.data.train_dl:
            learn.data.train_ds.actions = learn.predict(element)
            [c.on_step_end(learn=learn) for c in callbacks]

            counter += 1
            # if counter % 100 == 0:# or counter == 0:
        interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
        interp.plot_heatmapped_episode(epoch)

        [c.on_epoch_end() for c in callbacks]

        # # For now we are going to avoid executing callbacks here.
        # learn.model.eval()
        # for element in learn.data.valid_dl:
        #     learn.data.valid_ds.actions = learn.predict(element)

        # if epoch % 1 == 0:
        #     interp = AgentInterpretationAlpha(learn)
        #     interp.plot_heatmapped_episode(epoch)

    [c.on_train_end() for c in callbacks]


def test_interpretation_plot_q_dqn_returns():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 20

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, max_episodes=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(episode=epoch) for c in callbacks]
        learn.model.train()
        counter = 0
        for element in learn.data.train_dl:
            learn.data.train_ds.actions = learn.predict(element)
            [c.on_step_end(learn=learn) for c in callbacks]

            counter += 1
            # if counter % 100 == 0:# or counter == 0:
        if epoch % 5 == 0:
            interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
            interp.plot_q_density(epoch)

        [c.on_epoch_end() for c in callbacks]
    [c.on_train_end() for c in callbacks]


def test_interpretation_plot_q_ddpg_returns():
    data = MDPDataBunch.from_env('Pendulum-v0', render='human')
    # data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human')
    model = DDPG(data, batch=8)
    learn = AgentLearner(data, model)

    epochs = 20

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, max_episodes=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(episode=epoch) for c in callbacks]
        learn.model.train()
        for element in learn.data.train_dl:
            learn.data.train_ds.actions = learn.predict(element)

            [c.on_step_end(learn=learn) for c in callbacks]

        if epoch % 2 == 0:
            interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
            interp.plot_q_density(epoch)
        [c.on_epoch_end(learn=learn) for c in callbacks]
    [c.on_train_end() for c in callbacks]

