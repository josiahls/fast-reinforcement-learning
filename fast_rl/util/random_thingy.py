"""
from fast_rl.core.Interpreter import AgentInterpretationAlpha

interp = AgentInterpretationAlpha(learn)
interp.plot_heatmapped_episode(-1)

"""
from fastai.gen_doc.nbtest import doctest

doctest('FixedTargetDQN.__init__')