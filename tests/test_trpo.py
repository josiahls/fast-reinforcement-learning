from fast_rl.agents.old_trpo import TRPOLearner, TRPOTrainer
from fast_rl.agents.old_trpo_models import TRPOModule
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.core.data_block import MDPDataBunch


def test_trpo_learner_init_discrete():
	trpo_learner=TRPOLearner(
		data=MDPDataBunch.from_env('CartPole-v1',bs=8,add_valid=False),
		model=TRPOModule(4,2,0.001),
		memory=ExperienceReplay(100),
		trainers=TRPOTrainer
	)

def test_trpo_learner_init_continuous():
	trpo_learner=TRPOLearner(
		data=MDPDataBunch.from_env('AntPyBulletEnv-v0',bs=8,add_valid=False),
		model=TRPOModule(4,8,0.001,discrete=False),
		memory=ExperienceReplay(100),
		trainers=TRPOTrainer
	)


def test_trpo_learner_discrete_fit():
	trpo_learner=TRPOLearner(
		data=MDPDataBunch.from_env('CartPole-v1',bs=8,add_valid=False),
		model=TRPOModule(4,2,0.001),
		memory=ExperienceReplay(100),
		trainers=TRPOTrainer
	)
	trpo_learner.fit(4)


def test_trpo_model_calc_q():
	trpo_learner=TRPOLearner(
		data=MDPDataBunch.from_env('CartPole-v1',bs=8,add_valid=False),
		model=TRPOModule(4,2,0.001),
		memory=ExperienceReplay(100),
		trainers=TRPOTrainer
	)

def test_trpo_model_calc_v(): pass
def test_trpo_model_calc_adv(): pass
