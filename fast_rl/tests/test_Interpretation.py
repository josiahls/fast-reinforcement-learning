from typing import Collection

from fastai.basic_data import DatasetType

from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import DQN, FixedTargetDQN
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearnerAlpha
from fast_rl.core.MarkovDecisionProcess import MDPDataBunchAlpha, FEED_TYPE_IMAGE, FEED_TYPE_STATE


# def test_interpretation_heatmap():
#     data = MDPDataBunchAlpha.from_env('maze-random-5x5-v0', render='human', max_steps=100,
#                                  feed_type=FEED_TYPE_STATE, add_valid=False, memory_management_strategy='non')
#     model = DQN(data, batch_size=8)
#     learn = AgentLearnerAlpha(data, model)
#
#     learn.fit(5)
#     interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
#     interp.plot_heatmapped_episode(-1, action_index=0)


def test_interpretation_plot_q_dqn_returns():
    data = MDPDataBunchAlpha.from_env('maze-random-5x5-v0', max_steps=100, render='human', add_valid=False,
                                      memory_management_strategy='non')
    model = DQN(data)
    learn = AgentLearnerAlpha(data, model)
    learn.fit(5)
    interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
    interp.plot_heatmapped_episode(2)


# def test_inerpretation_plot_model_accuracy_fixeddqn():
#     data = MDPDataBunchAlpha.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False,
#                                  memory_management_strategy='non')
#     model = FixedTargetDQN(data, batch_size=64, max_episodes=100, copy_over_frequency=4)
#     learn = AgentLearnerAlpha(data, model)
#
#     learn.fit(5)
#     interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
#     interp.plot_agent_accuracy_density()


# def test_interpretation_plot_q_density():
#     data = MDPDataBunchAlpha.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False,
#                                  memory_management_strategy='non')
#     model = FixedTargetDQN(data, batch_size=128, max_episodes=100, copy_over_frequency=3, use_embeddings=True)
#     learn = AgentLearnerAlpha(data, model)
#
#     learn.fit(4)
#     interp = AgentInterpretationAlpha(learn, ds_type=DatasetType.Train)
#     interp.plot_agent_accuracy_density()
