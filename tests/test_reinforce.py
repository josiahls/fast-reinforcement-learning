import pytest

from fast_rl.agents.reinforce import BaseReinforceTrainer, ReinforceLearner
from fast_rl.agents.reinforce_models import PGN
from fast_rl.core.data_block import MDPDataBunch, partial, ResolutionWrapper
from fast_rl.core.metrics import *


def test_reinforce():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=1, add_valid=False, keep_env_open=False,
							   res_wrap=partial(ResolutionWrapper, w_step=4, h_step=4))
	model=PGN((4,),2)
	metrics=[RewardMetric, RollingRewardMetric,EpsilonMetric]
	learner=ReinforceLearner(data=data,model=model,trainers=BaseReinforceTrainer,episodes_to_train=4,
							 callback_fns=metrics,loss_func=lambda x,y:x)
	learner.fit(3,wd=0,lr=0.0001)


@pytest.mark.usefixtures('skip_performance_check')
def test_reinforce_perf():
	data=MDPDataBunch.from_env('CartPole-v0', render='rgb_array', bs=1, add_valid=False, keep_env_open=False,
							   res_wrap=partial(ResolutionWrapper, w_step=4, h_step=4))
	model=PGN((4,),2)
	metrics=[RewardMetric, RollingRewardMetric,EpsilonMetric]
	learner=ReinforceLearner(data=data,model=model,trainers=BaseReinforceTrainer,episodes_to_train=4,callback_fns=metrics,loss_func=lambda x,y:x)
	learner.fit(500,wd=0,lr=0.01)
