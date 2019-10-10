from typing import Collection

from fastai.basic_data import DatasetType

from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import DQN, FixedTargetDQN
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch, FEED_TYPE_IMAGE, FEED_TYPE_STATE


def test_interpretation_heatmap():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, feed_type=FEED_TYPE_STATE)
    model = DQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)
    interp = AgentInterpretationAlpha(learn)
    interp.plot_heatmapped_episode(-1, action_index=0)


def test_interpretation_plot_sequence():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000)
    model = DQN(data)
    learn = AgentLearner(data, model)

    epochs = 5

    callbacks = learn.model.learner_callbacks  # type: Collection[LearnerCallback]
    [c.on_train_begin(learn=learn, n_epochs=epochs) for c in callbacks]
    for epoch in range(epochs):
        [c.on_epoch_begin(epoch=epoch) for c in callbacks]
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
    [c.on_train_end() for c in callbacks]


def test_interpretation_plot_q_dqn_returns():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
    model = DQN(data)
    learn = AgentLearner(data, model)
    learn.fit(5)
    interp = AgentInterpretationAlpha(learn)
    interp.plot_heatmapped_episode(2)


def test_interpretation_plot_q_ddpg_returns():
    data = MDPDataBunch.from_env('Pendulum-v0', render='human', max_steps=100)
    # data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human')
    model = DDPG(data, batch=8)
    learn = AgentLearner(data, model)

    learn.fit(5)
    interp = AgentInterpretationAlpha(learn)
    interp.plot_q_density(-1)


def test_inerpretation_plot_model_accuracy_fixeddqn():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, batch_size=64, max_episodes=100, copy_over_frequency=4)
    learn = AgentLearner(data, model)

    learn.fit(5)
    interp = AgentInterpretationAlpha(learn)
    interp.plot_agent_accuracy_density()


def test_interpretation_plot_q_density():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000, add_valid=False)
    model = FixedTargetDQN(data, batch_size=128, max_episodes=100, copy_over_frequency=3)
    learn = AgentLearner(data, model)

    learn.fit(4)
    interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
    interp.plot_agent_accuracy_density()
