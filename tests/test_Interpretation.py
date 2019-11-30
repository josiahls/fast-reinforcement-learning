# from itertools import product
#
# import pytest
# from fastai.basic_data import DatasetType
#
# from fast_rl.agents.dqn import DQN
# from fast_rl.core.data_block import MDPDataBunch
# from fast_rl.core.basic_train import AgentLearner
# from fast_rl.core.train import AgentInterpretation, GroupAgentInterpretation
#
#
# reward_plot_list = list(product([True, False], repeat=2))
#
# def test_interpretation_reward_group_plot():
#     group_interp = GroupAgentInterpretation()
#     for i in range(2):
#         data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=4, add_valid=False)
#         model = DQN(data)
#         learn = AgentLearner(data, model)
#         learn.fit(2)
#
#         interp = AgentInterpretation(learn=learn, ds_type=DatasetType.Train)
#         interp.plot_rewards(cumulative=True, per_episode=True, group_name='run1')
#         group_interp.add_interpretation(interp)
#
#     group_interp.plot_reward_bounds(return_fig=True, per_episode=True).show()
#
# @pytest.mark.parametrize(["per_episode", "cumulative"], reward_plot_list)
# def test_interpretation_reward_plot(per_episode, cumulative):
#     data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array')
#     model = DQN(data)
#     learn = AgentLearner(data, model)
#     learn.fit(2)
#
#     interp = AgentInterpretation(learn=learn)
#     interp.plot_rewards(return_fig=True, per_episode=per_episode, cumulative=cumulative).show()
#     data.train_ds.env.close()
#     data.valid_ds.env.close()
