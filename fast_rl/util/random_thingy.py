"""
from fast_rl.core.Interpreter import AgentInterpretationAlpha

interp = AgentInterpretationAlpha(learn)
interp.plot_heatmapped_episode(-1)

"""
from fast_rl.core.basic_train import AgentLearner
from fast_rl.agents.DQN import FixedTargetDQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay

data = MDPDataBunch.from_env('Pong-v0', render='human', max_steps=100, add_valid=False)
model = FixedTargetDQN(data, memory=ExperienceReplay(memory_size=100000, reduce_ram=True))
learn = AgentLearner(data, model)
learn.fit(450)