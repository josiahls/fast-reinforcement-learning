from collections import Collection

from fastai.basic_train import LearnerCallback

from fast_rl.agents.DDPG import DDPG
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


def test_ddpg():
    data = MDPDataBunch.from_env('Pendulum-v0', render='human')
    # data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human')
    model = DDPG(data, batch=8)
    learn = AgentLearner(data, model)

    epochs = 450

    callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, max_episodes=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(episode=epoch) for c in callbacks]
        learn.model.train()
        for element in learn.data.train_dl:
            learn.data.train_ds.actions = learn.predict(element)

            [c.on_step_end(learn=learn) for c in callbacks]
        [c.on_epoch_end(learn=learn) for c in callbacks]
    [c.on_train_end() for c in callbacks]