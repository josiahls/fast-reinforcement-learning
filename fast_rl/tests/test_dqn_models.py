from fast_rl.agents.DQN import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon
from fast_rl.core.basic_train import AgentLearner


def test_basic_dqn_model_maze():
    print('\n')
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = DQN(data)
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn


def test_fixed_target_dqn_model_maze():
    print('\n')
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = FixedTargetDQN(data)
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn

def test_fixed_target_dqn_model_cartpole():
    print('\n')
    data = MDPDataBunch.from_env('CartPole-v1', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, memory=ExperienceReplay(memory_size=100000, reduce_ram=True))
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn


def test_fixed_target_dqn_no_explore_model_maze():
    print('\n')
    data = MDPDataBunch.from_env('CartPole-v1', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, exploration_strategy=GreedyEpsilon(epsilon_start=0, epsilon_end=0,
                                                                    decay=0.001, do_exploration=False))
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn


def test_double_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', bs=8, max_steps=100)
    model = DoubleDQN(data)
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn


def test_dueling_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', bs=8, max_steps=100)
    model = DuelingDQN(data)
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn


def test_double_dueling_dqn_model_maze():
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = DoubleDuelingDQN(data)
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn
