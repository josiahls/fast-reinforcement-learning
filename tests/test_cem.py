import pytest
from fastai.tabular.data import emb_sz_rule

from fast_rl.agents.cem import CEMLearner, CEMTrainer
from fast_rl.agents.cem_models import CEMModel
from fast_rl.core.data_block import MDPDataBunch
import numpy as np

from fast_rl.core.metrics import RewardMetric, RollingRewardMetric


@pytest.mark.usefixtures('skip_performance_check')
def test_cem():
	data=MDPDataBunch.from_env('CartPole-v0',render='human',add_valid=False,bs=16)
	bs, state, action=data.bs, data.state, data.action
	if np.any(state.n_possible_values==np.inf):
		emb_szs=[]
	else:
		emb_szs=[(d+1, int(emb_sz_rule(d))) for d in state.n_possible_values.reshape(-1, )]

	model=CEMModel(4,2,embed_szs=emb_szs,layers=[128])
	reinforce_learner=CEMLearner(data,model,trainers=[CEMTrainer],callback_fns=[RewardMetric,RollingRewardMetric])
	reinforce_learner.fit(600,lr=0.01,wd=0)
