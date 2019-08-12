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

## Roadmap (kind of)
At the moment these are the things I personally urgently need, and then the nice things that will make this repo
something akin to valuable. These are listed in kind of the order I am planning on executing them.

**Critical**
- [X] MDPDataBunch: Finished to the point of being useful. Please reference: `tests/test_Envs`
Example:
```python
from fast_rl.core.Envs import Envs
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch

# At present will try to load OpenAI, box2d, pybullet, atari, maze.
# note "get_all_latest_envs" has a key inclusion and exclusion so if you don't have some of these envs installed, 
# you can avoid this here.
for env in Envs.get_all_latest_envs():
    max_steps = 50  # Limit the number of per episode iterations for now.
    print(f'Testing {env}')
    mdp_databunch = MDPDataBunch.from_env(env, max_steps=max_steps, num_workers=0)
    if mdp_databunch is None:
        print(f'Env {env} is probably Mujoco... Add imports if you want and try on your own. Don\'t like '
              f'proprietary engines like this. If you have any issues, feel free to make a PR!')
    else:
        epochs = 1 # N episodes to run
        for epoch in range(epochs):
            for state in mdp_databunch.train_dl:
                # Instead of random action, you would have your agent here
                mdp_databunch.train_ds.actions = mdp_databunch.train_ds.get_random_action()
    
            for state in mdp_databunch.valid_dl:
                # Instead of random action, you would have your agent here and have exploration to 0
                mdp_databunch.valid_ds.actions = mdp_databunch.valid_ds.get_random_action()
```
- [X] DQN Agent: Reference `tests/test_Learner/test_basic_dqn_model_maze`. This test is
kind of a hell-scape. You will notice I plan to use Learner callbacks for a fit function. Also note, the gym_maze envs
will be important for at least discrete testing because you can heatmap the maze with the model's rewards. 
DQN Agent basic learning / optimization is done. It is undoubtedly unstable / buggy. Please note the next step.

One of the biggest issues with basic DQNs is the fact that Q values are often always moving. The actual basic DQN should
be a fixed targeting DQN, however lets move to some debugging tools so we are more effactive.

Testable code:
```python
from fast_rl.agents.DQN import DQN
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch

data = MDPDataBunch.from_env('maze-random-5x5-v0', render='human')
model = DQN(data)
learn = AgentLearner(data, model)

epochs = 450

callbacks = learn.model.callbacks  # type: Collection[LearnerCallback]
[c.on_train_begin(max_episodes=epochs) for c in callbacks]
for epoch in range(epochs):
    [c.on_epoch_begin(episode=epoch) for c in callbacks]
    learn.model.train()
    for element in learn.data.train_dl:
        learn.data.train_ds.actions = learn.predict(element)

        [c.on_step_end(learn=learn) for c in callbacks]
    [c.on_epoch_end() for c in callbacks]
[c.on_train_end() for c in callbacks]
``` 
Result:

![](res/pre_interpretation_maze_dqn.gif)

I believe that the agent explodes after the first episode. Not to worry! We will make a RL interpreter to see whats 
going on!

- [ ] **Working On** ReinforcementInterpretation: First method will be heatmapping the image / state space of the 
environment with the expected rewards for super important debugging. In the code above, we are testing with a maze for a
good reason. Heatmapping rewards over a maze is pretty easy as opposed to other environments.

- [ ] Learner Basic: After DQN and adding DDQN, Fixed targeting, DDDQN, we need to convert this (most likely) messy test
into a suitable object. Will be similar to the basic learner.
- [ ] DDPG Agent: We need to have at least one agent able to perform continuous environment execution. As a note, we 
could give discrete agents the ability to operate in a continuous domain via binning. 
- [ ] Learner Refactor: DDPG will probably screw up everything lol. We will need to rethink the learner / maybe try to
eliminate some custom methods for native Fastai library methods. 
**Additional**
- [ ] Single Global fit function like Fastai's. Better yet, actually just use their universal fit function.


## Code 
Some of the key take aways is Fastai's use of callbacks. Not only do callbacks allow for logging, but in fact adding a
callback to a generic fit function can change its behavior drastically. My goal is to have a library that is as easy
as possible to run on a server or on one's own computer. I am also interested in this being easy to extend. 

I have a few assumptions that the code / support algorithms I believe should adhere to:
- Environments should be pickle-able, and serializable. They should be able to shut down and start up multiple times
during run time.
- Agents should not need more information than images or state values for an environment per step. This means that 
environments should not be expected to allow output of contact points, sub-goals, or STRIPS style logical outputs. 

Rational:
- Shutdown / Startup: Some environments (pybullet) have the issue of shutting down and starting different environments.
Luckily, I have a fork of pybullet, so these modifications will be forced. 
- Pickling: Being able to encapsulate an environment as a `.pkl` can be important for saving it and all the information
it generated.
- Serializable: If we want to do parallel processing, environments need to be serializable to transport them between 
those processes.

Some extra assumptions:
- Environments can easier be goal-less, or have a single goal in which OpenAI defines as `Env` and `GoalEnv`. 

These assumptions are necessary for us to implement other envs from other repos. We do not want to be tied to just
OpenAI gyms. 

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
