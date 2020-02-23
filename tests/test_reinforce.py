# from fastai.core import ifnone
#
# from fast_rl.agents.reinforce import REINFORCELearner
# from fast_rl.agents.reinforce_models import REINFORCEModel
# from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_STATE
#
#
# def test_reinforce_fit():
# 	data=MDPDataBunch.from_env('CartPole-v1')
# 	model=REINFORCEModel(4,2)
# 	reinforce_learner=REINFORCELearner(data,model)
# 	reinforce_learner.fit(3)