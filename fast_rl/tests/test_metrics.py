from fast_rl.agents.dqn import FixedTargetDQN
from fast_rl.core.agent_core import ExperienceReplay
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.metrics import RewardMetric


def test_metrics_reward_init():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', bs=4, max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, memory=ExperienceReplay(1000))
    learn = AgentLearner(data, model, callback_fns=[RewardMetric])
    learn.fit(3)