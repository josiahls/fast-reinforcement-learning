[![Build Status](https://dev.azure.com/jokellum/jokellum/_apis/build/status/josiahls.fast-reinforcement-learning?branchName=master)](https://dev.azure.com/jokellum/jokellum/_build/latest?definitionId=1&branchName=master)
[![pypi fasti_rl version](https://img.shields.io/pypi/v/fast_rl)](https://pypi.python.org/pypi/fast_rl)
[![github_master version](https://img.shields.io/github/v/release/josiahls/fast-reinforcement-learning?include_prereleases)](https://github.com/josiahls/fast-reinforcement-learning/releases)

**Note: Test passing will not be a useful stability indicator until version 1.0+**

# Fast Reinforcement Learning
This repo is not affiliated with Jeremy Howard or his course which can be found here: [here](https://www.fast.ai/about/)
We will be using components from the Fastai library however for building and training our reinforcement learning (RL) 
agents.

As a note, here is a run down of existing RL frameworks:
- [Intel Coach](https://github.com/NervanaSystems/coach) 
- [Tensor Force](https://github.com/tensorforce/tensorforce)
- [OpenAI Baselines](https://github.com/openai/baselines)
- [Tensorflow Agents](https://github.com/tensorflow/agents)
- [KerasRL](https://github.com/keras-rl/keras-rl)

However there are also frameworks in PyTorch most notably Facebook's Horizon:
- [Horizon](https://github.com/facebookresearch/Horizon)
- [DeepRL](https://github.com/ShangtongZhang/DeepRL)

Fastai for computer vision and tabular learning has been amazing. One would wish that this would be the same for RL. 
The purpose of this repo is to have a framework that is as easy as possible to start, but also designed for testing
new agents. 

# Table of Contents
1. [Installation](#installation)
2. [Alpha TODO](#alpha-todo)
3. [Code](#code)
5. [Versioning](#versioning)
6. [Contributing](#contributing)
7. [Style](#style)


## Installation
Very soon we would like to add some form of scripting to install some complicated dependencies. We have 2 steps:

**1.a FastAI**
[Install Fastai](https://github.com/fastai/fastai/blob/master/README.md#installation)
or if you are Anaconda (which is a good idea to use Anaconda) you can do: \
`conda install -c pytorch -c fastai fastai`


**1.b Optional / Extra Envs**
OpenAI all gyms: \
`pip install gym[all]`

Mazes: \
`git clone https://github.com/MattChanTK/gym-maze.git` \
`cd gym-maze` \
`python setup.py install`


**2 Actual Repo** \
`git clone https://github.com/josiahls/fast-reinforcement-learning.git` \
`cd fast-reinforcement-learning` \
`python setup.py install`

## Alpha TODO
At the moment these are the things we personally urgently need, and then the nice things that will make this repo
something akin to valuable. These are listed in kind of the order we are planning on executing them.

At present, we are in the Alpha stages of agents not being fully tested / debugged. The final step (before 1.0.0) will 
be doing an evaluation of the DQN and DDPG agent implementations and verifying each performs the best it can at 
known environments. Prior to 1.0.0, new changes might break previous code versions, and models are not guaranteed to be
working at their best. Post 1.0.0 will be more formal feature development with CI, unit testing, etc. 

**Critical**
Testable code:
```python
from fast_rl.agents.dqn import *
from fast_rl.agents.dqn_models import *
from fast_rl.core.agent_core import ExperienceReplay, GreedyEpsilon
from fast_rl.core.data_block import MDPDataBunch
from fast_rl.core.metrics import *

data = MDPDataBunch.from_env('CartPole-v1', render='rgb_array', bs=32, add_valid=False)
model = create_dqn_model(data, FixedTargetDQNModule, opt=torch.optim.RMSprop, lr=0.00025)
memory = ExperienceReplay(memory_size=1000, reduce_ram=True)
exploration_method = GreedyEpsilon(epsilon_start=1, epsilon_end=0.1, decay=0.001)
learner = dqn_learner(data=data, model=model, memory=memory, exploration_method=exploration_method)
learner.fit(10)
```

- [X] 0.7.0 Full test suite using multi-processing. Connect to CI.
- [X] 0.8.0 Comprehensive model eval **debug/verify**. Each model should succeed at at least a few known environments. Also, massive refactoring will be needed.
- [X] 0.9.0 Notebook demonstrations of basic model usage.
- [ ] **Working on** **1.0.0** Base version is completed with working model visualizations proving performance / expected failure. At 
this point, all models should have guaranteed environments they should succeed in. 
- [ ] 1.8.0 Add PyBullet Fetch Environments
    - [ ] 1.8.0 Not part of this repo, however the envs need to subclass the OpenAI `gym.GoalEnv`
    - [ ] 1.8.0 Add HER


## Code 
Some of the key take aways is Fastai's use of callbacks. Not only do callbacks allow for logging, but in fact adding a
callback to a generic fit function can change its behavior drastically. My goal is to have a library that is as easy
as possible to run on a server or on one's own computer. We are also interested in this being easy to extend. 

We have a few assumptions that the code / support algorithms I believe should adhere to:
- Environments should be pickle-able, and serializable. They should be able to shut down and start up multiple times
during run time.
- Agents should not need more information than images or state values for an environment per step. This means that 
environments should not be expected to allow output of contact points, sub-goals, or STRIPS style logical outputs. 

Rational:
- Shutdown / Startup: Some environments (pybullet) have the issue of shutting down and starting different environments.
Luckily, we have a fork of pybullet, so these modifications will be forced. 
- Pickling: Being able to encapsulate an environment as a `.pkl` can be important for saving it and all the information
it generated.
- Serializable: If we want to do parallel processing, environments need to be serializable to transport them between 
those processes.

Some extra assumptions:
- Environments can easier be goal-less, or have a single goal in which OpenAI defines as `Env` and `GoalEnv`. 

These assumptions are necessary for us to implement other envs from other repos. We do not want to be tied to just
OpenAI gyms. 

## Versioning
At present the repo is in alpha stages being. We plan to move this from alpha to a pseudo beta / working versions. 
Regardless of version, we will follow Python style versioning

_Alpha Versions_:  #.#.# e.g. 0.1.0. Alpha will never go above 0.99.99, at that point it will be full version 1.0.0.
                   A key point is during alpha, coding will be quick and dirty with no promise of proper deprecation.

_Beta / Full Versions_: These will be greater than 1.0.0. We follow the Python method of versions:
                        **[Breaking Changes]**.**[Backward Compatible Features]**.**[Bug Fixes]**. These will be feature
                        additions such new functions, tools, models, env support. Also proper deprecation will be used.
                        
_Pip update frequency_: We have a pip repository, however we do not plan to update it as frequently at the moment. 
                        However, the current frequency will be during Beta / Full Version updates, we might every 0.5.0
                        versions update pip.

## Contributing
Follow the templates we have on github. Make a branch either from master or the most recent version branch.
We recommend squashing commits / keep pointless ones to a minimum.


## Style
Fastai does not follow closely with [google python style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#3164-guidelines-derived-from-guidos-recommendations),
however in this repo we will use this guide.  
Some exceptions however (typically found in Fastai):
- "PEP 8 Multiple statements per line violation" is allowed in the case of if statements as long as they are still 
within the column limit.
