from fastai.basic_train import LearnerCallback
from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DQN import DQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch

t = tabular_learner()
def test_basic_dqn_model_maze():
    msg = 'the datasets in the dataloader seem to be different from the data bunches datasets...'

    data = MDPDataBunch.from_env('maze-random-10x10-plus-v0')
    model = DQN(data)

    epochs = 450

    callbacks = [model.callbacks]  # type: Collection[LearnerCallback]
    [c.on_train_begin() for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin() for c in callbacks]
        model.train()
        for element in data.train_dl:
            data.train_ds.actions = model(element)
            # print(f'state {element} action {mdp_databunch.train_dl.dl.dataset.actions}')
            assert np.sum(np.equal(data.train_dl.dl.dataset.actions, data.train_ds.actions)) == \
                np.size(data.train_ds.actions), msg
            [c.on_step_end() for c in callbacks]
        [c.on_epoch_end() for c in callbacks]

        # For now we are going to avoid executing callbacks here.
        model.eval()
        for element in data.valid_dl:
            data.valid_ds.actions = model(element)
            # print(f'state {element} action {mdp_databunch.valid_dl.dl.dataset.actions}')
            assert np.sum(np.equal(data.train_dl.dl.dataset.actions, data.train_ds.actions)) == \
                np.size(data.train_ds.actions), msg
    [c.on_train_end() for c in callbacks]
