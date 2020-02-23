from fastai.core import ifnone, np
from fastai.tabular.data import emb_sz_rule

from fast_rl.agents.reinforce import REINFORCELearner, REINFORCEStepWiseTrainer, REINFORCEEpisodicTrainer
from fast_rl.agents.reinforce_models import REINFORCEModel
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_STATE, partial, ResolutionWrapper


def test_reinforce_fit_step_wise():
	data=MDPDataBunch.from_env('maze-random-5x5-v0',max_steps=100,render='human',k=0, res_wrap=partial(ResolutionWrapper, w_step=3, h_step=3),add_valid=False)
	bs, state, action=data.bs, data.state, data.action
	if np.any(state.n_possible_values==np.inf):
		emb_szs=[]
	else:
		emb_szs=[(d+1, int(emb_sz_rule(d))) for d in state.n_possible_values.reshape(-1, )]

	model=REINFORCEModel(2,2,embed_szs=emb_szs)
	reinforce_learner=REINFORCELearner(data,model,trainers=[REINFORCEStepWiseTrainer])
	reinforce_learner.fit(450,lr=0.005)

def test_reinforce_fit_episodic():
	data=MDPDataBunch.from_env('maze-random-5x5-v0',max_steps=100,render='human',k=0,
		res_wrap=partial(ResolutionWrapper, w_step=3, h_step=3),add_valid=False,device='cpu')
	bs, state, action=data.bs, data.state, data.action
	if np.any(state.n_possible_values==np.inf):
		emb_szs=[]
	else:
		emb_szs=[(d+1, int(emb_sz_rule(d))) for d in state.n_possible_values.reshape(-1, )]

	model=REINFORCEModel(2,2,embed_szs=emb_szs)
	reinforce_learner=REINFORCELearner(data,model,trainers=[REINFORCEEpisodicTrainer])
	reinforce_learner.fit(450,lr=0.005)