[![Build Status](https://dev.azure.com/jokellum/jokellum/_apis/build/status/josiahls.fast-reinforcement-learning?branchName=master)](https://dev.azure.com/jokellum/jokellum/_build/latest?definitionId=1&branchName=master)
[![pypi fasti_rl version](https://img.shields.io/pypi/v/fast_rl)](https://pypi.python.org/pypi/fast_rl)
[![github_master version](https://img.shields.io/github/v/release/josiahls/fast-reinforcement-learning?include_prereleases)](https://github.com/josiahls/fast-reinforcement-learning/releases)

# Fast_rl
This repo is not affiliated with Jeremy Howard or his course which can be found [here](https://www.fast.ai/about/).
We will be using components from the Fastai library for building and training our reinforcement learning (RL) 
agents.

Our goal is for fast_rl to be make benchmarking easier, inference more efficient, and environment compatibility to be
as decoupled as much as possible. This being version 1.0, we still have a lot of work to make RL training itself faster 
and more efficient. The goals for this repo can be seen in the [RoadMap](#roadmap).

**An important note is that training can use up a lot of RAM. This will likely be resolved as more models are being added. Likely will be resolved by off loading to storage in the next few versions.**

A simple example:
```python
from fast_rl.agents.dqn import create_dqn_model, dqn_learner
from fast_rl.agents.dqn_models import *
from fast_rl.core.agent_core import ExperienceReplay,  GreedyEpsilon
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.metrics import RewardMetric, EpsilonMetric

memory = ExperienceReplay(memory_size=1000000, reduce_ram=True)
explore = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
data = MDPDataBunch.from_env('CartPole-v1', render='human', bs=64, add_valid=False)
model = create_dqn_model(data=data, base_arch=FixedTargetDQNModule, lr=0.001, layers=[32,32])
learn = dqn_learner(data, model, memory=memory, exploration_method=explore, copy_over_frequency=300,
                    callback_fns=[RewardMetric, EpsilonMetric])
learn.fit(450)
```

More complex examples might involve running an RL agent multiple times, generating episode snapshots as gifs, grouping
reward plots, and finally showing the best and worst runs in a single graph. 
```python
from fastai.basic_data import DatasetType
from fast_rl.agents.dqn import create_dqn_model, dqn_learner
from fast_rl.agents.dqn_models import *
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.metrics import RewardMetric, EpsilonMetric
from fast_rl.core.train import GroupAgentInterpretation, AgentInterpretation

group_interp = GroupAgentInterpretation()
for i in range(5):
	memory = ExperienceReplay(memory_size=1000000, reduce_ram=True)
	explore = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
	data = MDPDataBunch.from_env('CartPole-v1', render='human', bs=64, add_valid=False)
	model = create_dqn_model(data=data, base_arch=FixedTargetDQNModule, lr=0.001, layers=[32,32])
	learn = dqn_learner(data, model, memory=memory, exploration_method=explore, copy_over_frequency=300,
						callback_fns=[RewardMetric, EpsilonMetric])
	learn.fit(450)

	interp=AgentInterpretation(learn, ds_type=DatasetType.Train)
	interp.plot_rewards(cumulative=True, per_episode=True, group_name='cartpole_experience_example')
	group_interp.add_interpretation(interp)
	group_interp.to_pickle(f'{learn.model.name.lower()}/', f'{learn.model.name.lower()}')
	for g in interp.generate_gif(): g.write(f'{learn.model.name.lower()}')
group_interp.plot_reward_bounds(per_episode=True, smooth_groups=10)
```
More examples can be found in `docs_src` and the actual code being run for generating gifs can be found in `tests` in 
either `test_dqn.py` or `test_ddpg.py`.

As a note, here is a run down of existing RL frameworks:
- [Intel Coach](https://github.com/NervanaSystems/coach) 
- [Tensor Force](https://github.com/tensorforce/tensorforce)
- [OpenAI Baselines](https://github.com/openai/baselines)
- [Tensorflow Agents](https://github.com/tensorflow/agents)
- [KerasRL](https://github.com/keras-rl/keras-rl)

However there are also frameworks in PyTorch:
- [Horizon](https://github.com/facebookresearch/Horizon)
- [DeepRL](https://github.com/ShangtongZhang/DeepRL)
- [Spinning Up](https://spinningup.openai.com/en/latest/user/introduction.html)

## Installation

**fastai (semi-optional)**\
[Install Fastai](https://github.com/fastai/fastai/blob/master/README.md#installation)
or if you are using Anaconda (which is a good idea to use Anaconda) you can do: \
`conda install -c pytorch -c fastai fastai`

**fast_rl**\
Fastai will be installed if it does not exist. If it does exist, the versioning should be repaired by the the setup.py.
`pip install fastai`

## Installation (Optional)
OpenAI all gyms: \
`pip install gym[all]`

Mazes: \
`git clone https://github.com/MattChanTK/gym-maze.git` \
`cd gym-maze` \
`python setup.py install`


## Installation Dev (Optional)
`git clone https://github.com/josiahls/fast-reinforcement-learning.git` \
`cd fast-reinforcement-learning` \
`python setup.py install`

## Installation Issues
Many issues will likely fall under [fastai installation issues](https://github.com/fastai/fastai/blob/master/README.md#installation-issues).

Any other issues are likely environment related. It is important to note that Python 3.7 is not being tested due to
an issue with Pyglet and gym do not working. This issue will not stop you from training models, however this might impact using
OpenAI environments. 

## RoadMap

- [ ] 1.1.0 More Traditional RL models
    - [X] Add Cross Entropy Method CEM
    - [X] NStep Experience replay
    - [X] Gaussian and Factored Gaussian Noise exploration replacement
    - [ ] Add RAINBOW DQN
    - [ ] **Working on** Add REINFORCE
    - [ ] **Working on** Add PPO
    - [ ] **Working on** Add TRPO
    - [ ] Add D4PG
    - [ ] Add A2C
    - [ ] Add A3C
- [ ] 1.2.0 HRL models *Possibly might change version to 2.0 depending on SMDP issues*
    - [ ] Add SMDP
    - [ ] Add Goal oriented MDPs. Will Require a new "Step"
    - [ ] Add FeUdal Network
    - [ ] Add storage based DataBunch memory management. This can prevent RAM from being used up by episode image frames
    that may or may not serve any use to the agent, but only for logging.
- [ ] 1.3.0
    - [ ] Add HAC
    - [ ] Add MAXQ
    - [ ] Add HIRO
- [ ] 1.4.0
    - [ ] Add h-DQN
    - [ ] Add Modulated Policy Hierarchies
    - [ ] Add Meta Learning Shared Hierarchies
- [ ] 1.5.0
    - [ ] Add STRategic Attentive Writer (STRAW)
    - [ ] Add H-DRLN
    - [ ] Add Abstract Markov Decision Process (AMDP)
    - [ ] Add conda integration so that installation can be truly one step.
- [ ] 1.6.0 HRL Options models *Possibly will already be implemented in a previous model*
    - [ ] Options augmentation to DQN based models
    - [ ] Options augmentation to actor critic models
    - [ ] Options augmentation to async actor critic models
- [ ] 1.8.0 HRL Skills
    - [ ] Skills augmentation to DQN based models
    - [ ] Skills augmentation to actor critic models
    - [ ] Skills augmentation to async actor critic models
- [ ] 1.9.0 Add PyBullet Fetch Environments
    - [ ] Envs need to subclaNot part of this repo, however the ess the OpenAI `gym.GoalEnv`
    - [ ] Add HER
- [ ] 2.0.0 Breaking refactor of all methods
    - [ ] Environment needs to be faster. Beat openai baseline 350 frames per second.
    - [ ] Unify common code pieces shared in all models


## Contribution
Following fastai's guidelines would be desirable: [Guidelines](https://github.com/fastai/fastai/blob/master/README.md#contribution-guidelines)

While we hope that model additions will be added smoothly. All models will only be dependent on `core.layers.py`.
As time goes on, the model architecture will overall improve (we are and while continue to be still figuring things out).


## Style
Since fastai uses a different style from traditional PEP-8, we will be following [Style](https://docs.fast.ai/dev/style.html) 
and [Abbreviations](https://docs.fast.ai/dev/abbr.html). Also we will use RL specific abbr.

|        | Concept | Abbr. | Combination Examples |
|:------:|:-------:|:-----:|:--------------------:|
| **RL** |  State  |  st   |                      |
|        | Action  |  acn  |                      |
|        | Bounds  |  bb   | Same as Bounding Box |

## Examples

### Reward Graphs

|                                            |       Model     | 
|:------------------------------------------:|:---------------:|
| ![01](./res/reward_plots/cartpole_dqn.png) |      DQN     |
| ![01](./res/reward_plots/cartpole_dueling.png) |  Dueling DQN     |
| ![01](./res/reward_plots/cartpole_double.png) |  Double DQN     |
| ![01](./res/reward_plots/cartpole_dddqn.png) |    DDDQN     |
| ![01](./res/reward_plots/cartpole_fixedtarget.png) |     Fixed Target DQN     |
| ![01](./res/reward_plots/lunarlander_dqn.png) |      DQN     |
| ![01](./res/reward_plots/lunarlander_dueling.png) |  Dueling DQN     |
| ![01](./res/reward_plots/lunarlander_double.png) |  Double DQN     |
| ![01](./res/reward_plots/lunarlander_dddqn.png) |    DDDQN     |
| ![01](./res/reward_plots/lunarlander_fixedtarget.png) |     Fixed Target DQN     |
| ![01](./res/reward_plots/ant_ddpg.png) |    DDPG    |
| ![01](./res/reward_plots/pendulum_ddpg.png) |    DDPG    |
| ![01](./res/reward_plots/halfcheetah_ddpg.png) |    DDPG    |


### Agent Stages

|      Model    |   Gif(Early)    |   Gif(Mid)    |   Gif(Late)     |
|:------------:|:------------:|:------------:|:------------:|
| DDPG+PER | ![](./res/run_gifs/pendulum_PriorityExperienceReplay_DDPGModule_1_episode_35.gif)  | ![](./res/run_gifs/pendulum_PriorityExperienceReplay_DDPGModule_1_episode_222.gif)  | ![](./res/run_gifs/pendulum_PriorityExperienceReplay_DDPGModule_1_episode_431.gif)|
| DoubleDueling+ER | ![](./res/run_gifs/lunarlander_ExperienceReplay_DoubleDuelingModule_1_episode_114.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DoubleDuelingModule_1_episode_346.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DoubleDuelingModule_1_episode_925.gif)|
| DoubleDQN+ER | ![](./res/run_gifs/lunarlander_ExperienceReplay_DoubleDQNModule_1_episode_88.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DoubleDQNModule_1_episode_613.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DoubleDQNModule_1_episode_999.gif)|
| DuelingDQN+ER | ![](./res/run_gifs/lunarlander_ExperienceReplay_DuelingDQNModule_1_episode_112.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DuelingDQNModule_1_episode_431.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DuelingDQNModule_1_episode_980.gif)|
| DoubleDueling+PER | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DoubleDuelingModule_1_episode_151.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DoubleDuelingModule_1_episode_341.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DoubleDuelingModule_1_episode_999.gif)|
| DQN+ER | ![](./res/run_gifs/lunarlander_ExperienceReplay_DQNModule_1_episode_93.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DQNModule_1_episode_541.gif)  | ![](./res/run_gifs/lunarlander_ExperienceReplay_DQNModule_1_episode_999.gif)|
| DuelingDQN+PER | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DuelingDQNModule_1_episode_21.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DuelingDQNModule_1_episode_442.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DuelingDQNModule_1_episode_998.gif)|
| DQN+PER | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DQNModule_1_episode_99.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DQNModule_1_episode_382.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DQNModule_1_episode_949.gif)|
| DoubleDQN+PER | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DoubleDQNModule_1_episode_7.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DoubleDQNModule_1_episode_514.gif)  | ![](./res/run_gifs/lunarlander_PriorityExperienceReplay_DoubleDQNModule_1_episode_999.gif)|
| DDPG+PER | ![](./res/run_gifs/ant_PriorityExperienceReplay_DDPGModule_1_episode_52.gif)  | ![](./res/run_gifs/ant_PriorityExperienceReplay_DDPGModule_1_episode_596.gif)  | ![](./res/run_gifs/ant_PriorityExperienceReplay_DDPGModule_1_episode_984.gif)|
| DDPG+ER | ![](./res/run_gifs/ant_ExperienceReplay_DDPGModule_1_episode_54.gif)  | ![](./res/run_gifs/ant_ExperienceReplay_DDPGModule_1_episode_614.gif)  | ![](./res/run_gifs/ant_ExperienceReplay_DDPGModule_1_episode_999.gif)|
| DQN+PER | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DQNModule_1_episode_44.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DQNModule_1_episode_216.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DQNModule_1_episode_413.gif)|
| FixedTargetDQN+ER | ![](./res/run_gifs/cartpole_ExperienceReplay_FixedTargetDQNModule_1_episode_57.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_FixedTargetDQNModule_1_episode_309.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_FixedTargetDQNModule_1_episode_438.gif)|
| DQN+ER | ![](./res/run_gifs/cartpole_ExperienceReplay_DQNModule_1_episode_31.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DQNModule_1_episode_207.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DQNModule_1_episode_447.gif)|
| FixedTargetDQN+PER | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_FixedTargetDQNModule_1_episode_13.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_FixedTargetDQNModule_1_episode_265.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_FixedTargetDQNModule_1_episode_449.gif)|
| DoubleDQN+ER | ![](./res/run_gifs/cartpole_ExperienceReplay_DoubleDQNModule_1_episode_60.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DoubleDQNModule_1_episode_268.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DoubleDQNModule_1_episode_438.gif)|
| DoubleDQN+PER | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DoubleDQNModule_1_episode_35.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DoubleDQNModule_1_episode_269.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DoubleDQNModule_1_episode_444.gif)|
| DuelingDQN+ER | ![](./res/run_gifs/cartpole_ExperienceReplay_DuelingDQNModule_1_episode_62.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DuelingDQNModule_1_episode_209.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DuelingDQNModule_1_episode_432.gif)|
| DoubleDueling+PER | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DoubleDuelingModule_1_episode_2.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DoubleDuelingModule_1_episode_260.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DoubleDuelingModule_1_episode_438.gif)|
| DuelingDQN+PER | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DuelingDQNModule_1_episode_69.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DuelingDQNModule_1_episode_272.gif)  | ![](./res/run_gifs/cartpole_PriorityExperienceReplay_DuelingDQNModule_1_episode_438.gif)|
| DoubleDueling+ER | ![](./res/run_gifs/cartpole_ExperienceReplay_DoubleDuelingModule_1_episode_43.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DoubleDuelingModule_1_episode_287.gif)  | ![](./res/run_gifs/cartpole_ExperienceReplay_DoubleDuelingModule_1_episode_447.gif)|
| DDPG+ER | ![](./res/run_gifs/acrobot_ExperienceReplay_DDPGModule_1_episode_69.gif)  | ![](./res/run_gifs/acrobot_ExperienceReplay_DDPGModule_1_episode_197.gif)  | ![](./res/run_gifs/acrobot_ExperienceReplay_DDPGModule_1_episode_438.gif)|
| DDPG+PER | ![](./res/run_gifs/acrobot_PriorityExperienceReplay_DDPGModule_1_episode_55.gif)  | ![](./res/run_gifs/acrobot_PriorityExperienceReplay_DDPGModule_1_episode_267.gif)  | ![](./res/run_gifs/acrobot_PriorityExperienceReplay_DDPGModule_1_episode_422.gif)|