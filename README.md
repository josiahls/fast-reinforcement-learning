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

My motivation is that existing frameworks commonly use tensorflow, which nothing against tensorflow, but I have 
accomplished more in shorter periods of time using PyTorch. 

Fastai for computer vision and tabular learning has been amazing. One would wish that this would be the same for RL. 
The purpose of this repo is to have a framework that is as easy as possible to start, but also designed for testing
new agents. 

## Code 
Some of the key take aways is Fastai's use of callbacks. Not only do callbacks allow for logging, but in fact adding a
callback to a generic fit function can change its behavior drastically. My goal is to have a library that is as easy
as possible to run on a server or on ones own computer. I am also interested in this being easy to extend. 

I have a few assumptions that the code / support algorithms I believe should adhere to:
- Environments should be pickle-able, and serializable. They should be able to shut down and start up multiple times
during run time.
- Agents should not need more information than images or state values for an environment per step. This means that 
environments should not be expected to allow output of contact points, sub-goals, or STRIPS style logical outputs. 

Rational:
- Shutdown / Startup: Some environments (pybullet) have the issue of shutting down and starting different environments.
Luckily, I have a fork of pybullet, so this modifications will be forced. 
- Pickling: Being able to encapsulate an environment as a `.pkl` can be important for saving it and all the information
it generated.
- Serializable: If we want to do parallel processing, environments need to be serializable to transport them between 
those processes.

### Key Components

- DataBunch: First component needs to be done right, and well. We want to create a new type of DataBunch that 
actually represents an environment. The point here is to make human inference easier in jupyter notebooks. Not only that
we have an easy way to save a ran environment for later inference.

### 1. Preprocessing

### 2. Modeling

### 3. Dashboard

## TODO

### Git + Workflow


### Style
Fastai does not follow closely with [google python style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#3164-guidelines-derived-from-guidos-recommendations),
however in this repo we will use this guide.  
Some exceptions however (typically found in Fastai):
- "PEP 8 Multiple statements per line violation" is allowed in the case of if statements as long as they are still 
within the column limit.