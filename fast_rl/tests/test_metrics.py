from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
import fast_rl.core.Interpreter
from fast_rl.core.Interpreter import AgentInterpretationAlpha
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
import sys
import importlib


def test_epsilon():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, batch_size=64, max_episodes=100, copy_over_frequency=4)
    learn = AgentLearner(data, model)

    learn.fit(20)