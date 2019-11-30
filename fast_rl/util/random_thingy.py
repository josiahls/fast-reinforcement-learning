from functools import partial
from itertools import product

from fast_rl.agents.ddpg import DDPG
from fastai.basic_data import DatasetType

from fast_rl.agents.dqn import DQN, FixedTargetDQN, DoubleDQN, DuelingDQN, DoubleDuelingDQN
from fast_rl.core.agent_core import ExperienceReplay, PriorityExperienceReplay, GreedyEpsilon, OrnsteinUhlenbeck
from fast_rl.core.basic_train import AgentLearner, PipeLine
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_IMAGE, FEED_TYPE_STATE
from fast_rl.core.metrics import EpsilonMetric, RewardMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation
import torch


params_dqn = [DDPG]
params_experience = [ExperienceReplay, PriorityExperienceReplay]
params_state_format = [FEED_TYPE_STATE]  # , FEED_TYPE_IMAGE]

for model_cls, experience, s_format in product(params_dqn, params_experience, params_state_format):
    group_interp = GroupAgentInterpretation()
    for i in range(5):
        memory = experience(memory_size=1000000, reduce_ram=True)

        print('\n')
        data = MDPDataBunch.from_env('ReacherPyBulletEnv-v0', render='human', bs=64, add_valid=True,
                                     feed_type=s_format)

        model = partial(model_cls, memory=memory, opt=torch.optim.Adam,
                        exploration_strategy=OrnsteinUhlenbeck(size=data.action.taken_action.shape,
                                                               epsilon_start=1, epsilon_end=0.1,
                                                               decay=0.00001,
                                                               do_exploration=True))
        model = model(data)
        learn = AgentLearner(data, model, callback_fns=[RewardMetric, EpsilonMetric])
        learn.fit(2)

        meta = f'{experience.__name__}_{"FEED_TYPE_STATE" if s_format == FEED_TYPE_STATE else "FEED_TYPE_IMAGE"}'
        interp = AgentInterpretation(learn, ds_type=DatasetType.Train)
        interp.plot_rewards(cumulative=True, per_episode=True, group_name=meta)
        group_interp.add_interpretation(interp)
        group_interp.to_pickle(f'../../docs_src/data/mujocoreach_{model.name.lower()}/',
                               f'{model.name.lower()}_{meta}')