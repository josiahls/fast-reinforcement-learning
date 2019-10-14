
def test_basic_dqn_model_maze():
    from fast_rl.agents.DQN import DQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = DQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_fixed_target_dqn_model_maze():
    from fast_rl.agents.DQN import FixedTargetDQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
    print('\n')
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)

def test_fixed_target_dqn_model_cartpole():
    from fast_rl.agents.DQN import FixedTargetDQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
    from fast_rl.core.agent_core import ExperienceReplay
    print('\n')
    data = MDPDataBunch.from_env('CartPole-v1', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, memory=ExperienceReplay(memory_size=100000, reduce_ram=True))
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_fixed_target_dqn_no_explore_model_maze():
    from fast_rl.agents.DQN import FixedTargetDQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
    from fast_rl.core.agent_core import GreedyEpsilon
    print('\n')
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100, add_valid=False)
    model = FixedTargetDQN(data, lr=0.01, discount=0.8,
                           exploration_strategy=GreedyEpsilon(epsilon_start=0, epsilon_end=0,
                                                                               decay=0.001, do_exploration=False))
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_double_dqn_model_maze():
    from fast_rl.agents.DQN import DoubleDQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = DoubleDQN(data, batch_size=8)
    learn = AgentLearner(data, model)

    learn.fit(5)


def test_dueling_dqn_model_maze():
    from fast_rl.agents.DQN import DuelingDQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch

    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = DuelingDQN(data, batch_size=8)
    learn = AgentLearner(data, model)
    learn.fit(5)
    del learn


def test_double_dueling_dqn_model_maze():
    from fast_rl.agents.DQN import DoubleDuelingDQN
    from fast_rl.core.Learner import AgentLearner
    from fast_rl.core.MarkovDecisionProcess import MDPDataBunch

    data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', max_steps=100)
    model = DoubleDuelingDQN(data)
    learn = AgentLearner(data, model)

    learn.fit(5)
