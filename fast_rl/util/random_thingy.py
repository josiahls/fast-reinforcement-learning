from fast_rl.agents.dqn import DQN
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay
from fast_rl.core.basic_train import AgentLearner, PipeLine
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation


group_interp = GroupAgentInterpretation()

for i in range(5):
    data = MDPDataBunch.from_env('CartPole-v1', render='rgb_array', bs=128, device='cpu')
    model = DQN(data, memory=PriorityExperienceReplay(memory_size=100000, reduce_ram=True))
    learn = AgentLearner(data, model)
    learn.fit(450)
    interp = AgentInterpretation(learn)
    interp.plot_rewards(cumulative=True, per_episode=True, group_name='per', no_show=True)
    group_interp.add_interpretation(interp)
    group_interp.to_pickle('../../docs_src/data/dqn/', 'dqn_per')
    data.close()
group_interp.to_pickle('../../docs_src/data/dqn/', 'dqn_per')
"""
def pipeline_fn(num):
    group_interp = GroupAgentInterpretation()
    data = MDPDataBunch.from_env('CartPole-v1', render='rgb_array', bs=128, device='cpu')
    model = DQN(data, memory=PriorityExperienceReplay(memory_size=100000, reduce_ram=True))
    learn = AgentLearner(data, model)
    learn.fit(450)
    interp = AgentInterpretation(learn)
    interp.plot_rewards(cumulative=True, per_episode=True, group_name='per', no_show=True)
    group_interp.add_interpretation(interp)
    data.close()
    group_interp.to_pickle('../../docs_src/data/dqn', 'dqn' + str(num))
    return group_interp.analysis


pl = PipeLine(1, pipeline_fn)
print(pl.start(1))
"""
