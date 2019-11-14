from fastai.basic_data import DatasetType

from fast_rl.agents.dqn import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, GreedyEpsilon
from fast_rl.core.basic_train import AgentLearner, PipeLine
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_IMAGE
from fast_rl.core.metrics import EpsilonMetric, RewardMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation
import torch


for model_cls in [DoubleDuelingDQN]:#, DQN, FixedTargetDQN, DoubleDQN, DuelingDQN]:
    for meta in ['per_rms', 'er_rms']:
        group_interp = GroupAgentInterpretation()

        for i in range(5):
            data = MDPDataBunch.from_env('MountainCar-v0', render='rgb_array', bs=32, add_valid=False)
            #data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human', bs=32, device='cpu')

            if meta.__contains__('per_rms'): mem = PriorityExperienceReplay
            else: mem = ExperienceReplay

            if model_cls is DQN:
                model = model_cls(data, lr=0.001, layers=[64, 64], discount=0.99, grad_clip=1,
                                  memory=mem(memory_size=1000000, reduce_ram=True), optimizer=torch.optim.RMSprop,
                                  exploration_strategy=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.0001))
            else:
                model = model_cls(data, lr=0.001, layers=[64, 64], discount=0.99, grad_clip=1, tau=1.0,
                                         copy_over_frequency=300,
                                         memory=mem(memory_size=1000000, reduce_ram=True),
                                         optimizer=torch.optim.RMSprop,
                                  exploration_strategy=GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.0001))

            learn = AgentLearner(data, model, callback_fns=[EpsilonMetric, RewardMetric])
            learn.fit(450)
            interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
            interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
            group_interp.add_interpretation(interp)
            group_interp.to_pickle(f'../../docs_src/data/mountaincar_{model.name.lower()}/', f'{model.name.lower()}_{meta}')
            data.close()
        group_interp.to_pickle(f'../../docs_src/data/mountaincar_{model.name.lower()}/', f'{model.name.lower()}_{meta}')
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
