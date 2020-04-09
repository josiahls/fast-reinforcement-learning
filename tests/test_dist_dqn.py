from fast_rl.agents.dist_dqn import DistDQNLearner, BaseDistDQNTrainer
from fast_rl.agents.dist_dqn_models import DistributionalDQN
from fast_rl.core.data_block import MDPDataBunch, partial, ResolutionWrapper
from fast_rl.core.metrics import *


def test_dist_dqn():
	data=MDPDataBunch.from_env('CartPole-v0', render='human', bs=32, add_valid=False, keep_env_open=False,
							   res_wrap=partial(ResolutionWrapper, w_step=4, h_step=4))
	model=DistributionalDQN((4,),2)
	metrics=[RewardMetric, RollingRewardMetric,EpsilonMetric]
	learner=DistDQNLearner(data=data,model=model,trainers=BaseDistDQNTrainer,callback_fns=metrics,loss_func=lambda x,y:x)
	learner.fit(1600,wd=0,lr=0.0001)