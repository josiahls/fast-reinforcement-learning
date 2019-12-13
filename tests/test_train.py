from fastai.basic_data import DatasetType

from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.train import *
import pandas as pd

#
# def test_groupagentinterpretation_from_pickle():
#     group_interp = GroupAgentInterpretation.from_pickle('./data/cartpole_dqn',
#                                                         'dqn_PriorityExperienceReplay_FEED_TYPE_STATE')
#     group_interp.plot_reward_bounds(return_fig=True, per_episode=True, smooth_groups=5).show()
#
#
# def test_groupagentinterpretation_analysis():
#     group_interp = GroupAgentInterpretation.from_pickle('./data/cartpole_dqn',
#                                                         'dqn_PriorityExperienceReplay_FEED_TYPE_STATE')
#     assert isinstance(group_interp.analysis, list)
#     group_interp.in_notebook = True
#     assert isinstance(group_interp.analysis, pd.DataFrame)




#
# def test_interpretation_reward_group_plot():
#     group_interp = GroupAgentInterpretation()
#     group_interp2 = GroupAgentInterpretation()
#
#     for i in range(2):
#         data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=4, add_valid=False)
#         model = DQN(data)
#         learn = AgentLearner(data, model)
#         learn.fit(2)
#
#         interp = AgentInterpretation(learn=learn, ds_type=DatasetType.Train)
#         interp.plot_rewards(cumulative=True, per_episode=True, group_name='run1')
#         group_interp.add_interpretation(interp)
#         group_interp2.add_interpretation(interp)
#
#     group_interp.plot_reward_bounds(return_fig=True, per_episode=True).show()
#     group_interp2.plot_reward_bounds(return_fig=True, per_episode=True).show()
#
#     new_interp = group_interp.merge(group_interp2)
#     assert len(new_interp.groups) == len(group_interp.groups) + len(group_interp2.groups), 'Lengths do not match'
#     new_interp.plot_reward_bounds(return_fig=True, per_episode=True).show()