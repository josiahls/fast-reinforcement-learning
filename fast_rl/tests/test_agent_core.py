from fastai.basic_train import LearnerCallback, DatasetType
from fastai.callback import Callback
from fastai.tabular import tabular_learner
from fastai.vision import cnn_learner, models

import numpy as np
from traitlets import List
from typing import Collection

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.Interpreter import AgentInterpretationv1
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import PriorityExperienceReplay


def test_priority_experience_replay():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = FixedTargetDQN(data, memory=PriorityExperienceReplay(1000))

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
        interp = AgentInterpretationv1(learn, ds_type=DatasetType.Train)
        interp.plot_heatmapped_episode(epoch)

        [c.on_epoch_end(learn=learn) for c in callbacks]
    [c.on_train_end() for c in callbacks]