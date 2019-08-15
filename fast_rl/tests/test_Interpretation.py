from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DQN import DQN
from fast_rl.core.Interpreter import AgentInterpretationv1
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


def test_interpretation_heatmap():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 10

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(max_episodes=epochs) for c in callbacks]
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
            interp = AgentInterpretationv1(learn)
            interp.plot_heatmapped_episode(epoch)
    [c.on_train_end() for c in callbacks]

def test_interpretation_plot_sequence():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 10

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(max_episodes=epochs) for c in callbacks]
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
            interp = AgentInterpretationv1(learn)
            interp.plot_episode(epoch)

    [c.on_train_end() for c in callbacks]
