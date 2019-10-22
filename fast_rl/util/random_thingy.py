"""
from fast_rl.core.Interpreter import AgentInterpretationAlpha

interp = AgentInterpretationAlpha(learn)
interp.plot_heatmapped_episode(-1)

"""
from fastai.basic_data import DatasetType
from fast_rl.agents.DDPG import DDPG
from fast_rl.agents.DQN import FixedTargetDQN, DQN
from fast_rl.core.Learner import AgentLearnerAlpha, AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunchAlpha, MDPDataBunch

from fast_rl.core.agent_core import GreedyEpsilon, OrnsteinUhlenbeck, ExperienceReplay
from fast_rl.core.metrics import EpsilonMetric

data = MDPDataBunch.from_env('CartPole-v0')
model = FixedTargetDQN(data)
learner = AgentLearner(data=data, model=model)

learner.fit(450)