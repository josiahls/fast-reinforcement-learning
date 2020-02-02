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

- [ ] **Working on** **1.0.0** Base version is completed with working model visualizations proving performance / expected failure. At 
this point, all models should have guaranteed environments they should succeed in. 
- [ ] 1.1.0 More Traditional RL models
    - [ ] Add PPO
    - [ ] Add TRPO
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
- [ ] 1.9.0
- [ ] 2.0.0 Add PyBullet Fetch Environments
    - [ ] 2.0.0 Not part of this repo, however the envs need to subclass the OpenAI `gym.GoalEnv`
    - [ ] 2.0.0 Add HER


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

