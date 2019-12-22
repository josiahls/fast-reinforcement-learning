# import pytest
#
# from fast_rl.agents.dqn import *
# from fast_rl.agents.dqn_models import FixedTargetDQNModule
# from fast_rl.core.agent_core import *
# from fast_rl.core.data_block import *
# from fast_rl.core.train import *
#
# p_model = [FixedTargetDQNModule]
# p_exp = [ExperienceReplay]
# p_format = [FEED_TYPE_STATE]
#
# config_env_expectations = {
# 	'CartPole-v1': {'action_shape': (1, 2), 'state_shape': (1, 4)},
# 	'maze-random-5x5-v0': {'action_shape': (1, 4), 'state_shape': (1, 2)}
# }
#
#
# @pytest.mark.parametrize(["model_cls", "s_format", "mem"], list(product(p_model, p_format, p_exp)))
# def test_train_gym_maze_interpretation(model_cls, s_format, mem):
# 	success = False
# 	while not success:
# 		try:
# 			data = MDPDataBunch.from_env('maze-random-5x5-v0', render='rgb_array', bs=5, max_steps=50,
# 										 add_valid=False, feed_type=s_format)
# 			model = create_dqn_model(data, model_cls, opt=torch.optim.RMSprop)
# 			memory = mem(10000)
# 			exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
# 			learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
# 			learner.fit(1)
#
# 			interp = GymMazeInterpretation(learner, ds_type=DatasetType.Train)
# 			for i in range(-1, 4): interp.plot_heat_map(action=i)
#
# 			success = True
# 		except Exception as e:
# 			if not str(e).__contains__('Surface'):
# 				raise Exception
#
#
# @pytest.mark.parametrize(["model_cls", "s_format", "mem"], list(product(p_model, p_format, p_exp)))
# def test_train_q_value_interpretation(model_cls, s_format, mem):
# 	success = False
# 	while not success:
# 		try:
# 			data = MDPDataBunch.from_env('maze-random-5x5-v0', render='rgb_array', bs=5, max_steps=50,
# 										 add_valid=False, feed_type=s_format)
# 			model = create_dqn_model(data, model_cls, opt=torch.optim.RMSprop)
# 			memory = mem(10000)
# 			exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
# 			learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
# 			learner.fit(1)
#
# 			interp = QValueInterpretation(learner, ds_type=DatasetType.Train)
# 			interp.plot_q()
#
# 			success = True
# 		except Exception as e:
# 			if not str(e).__contains__('Surface'):
# 				raise Exception(e)
#
# #
# # def test_groupagentinterpretation_from_pickle():
# #     group_interp = GroupAgentInterpretation.from_pickle('./data/cartpole_dqn',
# #                                                         'dqn_PriorityExperienceReplay_FEED_TYPE_STATE')
# #     group_interp.plot_reward_bounds(return_fig=True, per_episode=True, smooth_groups=5).show()
# #
# #
# # def test_groupagentinterpretation_analysis():
# #     group_interp = GroupAgentInterpretation.from_pickle('./data/cartpole_dqn',
# #                                                         'dqn_PriorityExperienceReplay_FEED_TYPE_STATE')
# #     assert isinstance(group_interp.analysis, list)
# #     group_interp.in_notebook = True
# #     assert isinstance(group_interp.analysis, pd.DataFrame)
#
#
#
#
# #
# # def test_interpretation_reward_group_plot():
# #     group_interp = GroupAgentInterpretation()
# #     group_interp2 = GroupAgentInterpretation()
# #
# #     for i in range(2):
# #         data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=4, add_valid=False)
# #         model = DQN(data)
# #         learn = AgentLearner(data, model)
# #         learn.fit(2)
# #
# #         interp = AgentInterpretation(learn=learn, ds_type=DatasetType.Train)
# #         interp.plot_rewards(cumulative=True, per_episode=True, group_name='run1')
# #         group_interp.add_interpretation(interp)
# #         group_interp2.add_interpretation(interp)
# #
# #     group_interp.plot_reward_bounds(return_fig=True, per_episode=True).show()
# #     group_interp2.plot_reward_bounds(return_fig=True, per_episode=True).show()
# #
# #     new_interp = group_interp.merge(group_interp2)
# #     assert len(new_interp.groups) == len(group_interp.groups) + len(group_interp2.groups), 'Lengths do not match'
# #     new_interp.plot_reward_bounds(return_fig=True, per_episode=True).show()
