from itertools import product

import pytest

from fast_rl.agents.DQN import DQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.train import AgentInterpretation

reward_plot_list = list(product([True, False], repeat=3))


@pytest.mark.parametrize(["per_episode", "cumulative", "squash_episodes"], reward_plot_list)
def test_interpretation_reward_plot(per_episode, cumulative, squash_episodes):
    data = MDPDataBunch.from_env('CartPole-v0', render='rgb_array')
    model = DQN(data)
    learn = AgentLearner(data, model)
    learn.fit(2)

    interp = AgentInterpretation(learn=learn)
    interp.plot_rewards(return_fig=True, per_episode=per_episode, cumulative=cumulative, squash_episodes=squash_episodes).show()
    data.train_ds.env.close()
    data.valid_ds.env.close()
