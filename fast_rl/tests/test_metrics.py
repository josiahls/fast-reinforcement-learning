from fast_rl.agents.DQN import FixedTargetDQN
from fast_rl.core.Learner import AgentLearnerAlpha
from fast_rl.core.MarkovDecisionProcess import MDPDataBunchAlpha


def test_epsilon():
    data = MDPDataBunchAlpha.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = FixedTargetDQN(data, batch_size=8)
    learn = AgentLearnerAlpha(data, model)
    learn.fit(5)