"""
from fast_rl.core.Interpreter import AgentInterpretationAlpha

interp = AgentInterpretationAlpha(learn)
interp.plot_heatmapped_episode(-1)

"""
from fastai.basic_data import DatasetType
from fast_rl.agents.DDPG import DDPG
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch

# data = MDPDataBunch.from_env('Pendulum-v0', render='human')
from fast_rl.core.agent_core import GreedyEpsilon, OrnsteinUhlenbeck

data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000, add_valid=False)
# data = MDPDataBunch.from_env('Pendulum-v0', render='human', add_valid=False)
# data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human', add_valid=False)
model = DDPG(data, batch=128, lr=0.01, env_was_discrete=True,
             exploration_strategy=OrnsteinUhlenbeck(4, do_exploration=True))
learn = AgentLearner(data, model)
learn.fit(40)


from fast_rl.core.Interpreter import AgentInterpretationAlpha

interp = AgentInterpretationAlpha(learn, DatasetType.Train)
interp.plot_heatmapped_episode(-1)