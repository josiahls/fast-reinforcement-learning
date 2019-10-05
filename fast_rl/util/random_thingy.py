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
from fast_rl.core.agent_core import GreedyEpsilon, OrnsteinUhlenbeck, ExperienceReplay
from fast_rl.core.metrics import EpsilonMetric

# data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=1000, add_valid=False)
data = MDPDataBunch.from_env('Pendulum-v0', render='human', add_valid=False)
# data = MDPDataBunch.from_env('MountainCarContinuous-v0', render='human', add_valid=False)
model = DDPG(data, batch=128, memory=ExperienceReplay(100000, reduce_ram=True),
             exploration_strategy=OrnsteinUhlenbeck(epsilon_start=1, epsilon_end=0.1, decay=0.0001, size=1,
                                                    do_exploration=True, end_episode=450))
learn = AgentLearner(data, model, metrics=[EpsilonMetric])
learn.fit(4500)


from fast_rl.core.Interpreter import AgentInterpretationAlpha

interp = AgentInterpretationAlpha(learn, DatasetType.Train)
interp.plot_q_density(-1)