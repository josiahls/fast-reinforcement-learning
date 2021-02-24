import collections
from copy import deepcopy
from functools import partial
from warnings import warn

from fastai.basic_train import LearnerCallback
from fastai.imports import torch, Any

from fast_rl.agents.dist_dqn_models import TargetNet
from fast_rl.agents.dqn_models import distr_projection
from fast_rl.core.agent_core import ExperienceReplay, NStepExperienceReplay, NStepPriorityExperienceReplay, \
    ExplorationStrategy
from fast_rl.core.basic_train import AgentLearner, listify, List
from fast_rl.core.data_block import MDPDataBunch, MDPStep
from fastai.imports import torch

import gym
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import groupby


class PolicyExploration(ExplorationStrategy):
    def __init__(self,apply_softmax=False):
        super().__init__()
        self.apply_softmax=apply_softmax
        self.epsilon=0

    def perturb(self, action, action_space) -> np.ndarray:
        if self.apply_softmax:
            action=F.softmax(action.double(), dim=1) # cast to double so the precision is greater (otherwise sum!= 1 err)
        action/=action.sum()
        action_list = [np.random.choice(len(prob), p=prob) for prob in action]
        return np.array(action_list)

def calc_qvals(rewards,gamma):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    res = list(reversed(res))
    mean_q = np.mean(res)
    return [q - mean_q for q in res]

def calc_loss(states,actions,logits,q_vals):
    log_prob_v=F.log_softmax(logits, dim=1)
    log_prob_actions_v=q_vals*log_prob_v[range(len(states)), actions]
    loss_v=-log_prob_actions_v.mean()
    return loss_v

class BaseReinforceTrainer(LearnerCallback):
    def __init__(self, learn: 'ReinforceLearner', max_episodes=None):
        r"""Handles basic DQN end of step model optimization."""
        super().__init__(learn)

        self.n_skipped = 0
        self._persist = max_episodes is not None
        self.max_episodes = max_episodes
        self.episode = -1
        self.iteration = 0
        # For the callback handler
        self._order = 0
        self.previous_item = None
        self.loss=None

    @property
    def learn(self)->'ReinforceLearner':
        return self._learn()

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs if not self._persist else self.max_episodes

    def on_epoch_begin(self, epoch, **kwargs: Any):
        pass

    def on_backward_begin(self, **kwargs: Any):
        return {'skip_bwd':self.learn.current_n_episodes_to_train<self.learn.n_episodes_to_train or self.learn.warming_up}
    def on_backward_end(self, **kwargs:Any):
        if self.learn.current_n_episodes_to_train>=self.learn.n_episodes_to_train:
            return {'skip_step': False}
        return {'skip_step':True}
    def on_step_end(self, **kwargs: Any):
        if self.learn.current_n_episodes_to_train>=self.learn.n_episodes_to_train:
            self.learn.current_n_episodes_to_train=0
            return {'skip_zero': False}
        return {'skip_zero': True}

    def on_batch_begin(self, **kwargs:Any) ->None:
        if self.learn.model.training:
            if self.learn.data.x.items[-1].done: self.learn.current_n_episodes_to_train+=1
            if not self.learn.warming_up and self.learn.loss_func is None: self.learn.init_loss_func()

    def on_loss_begin(self, **kwargs: Any):
        r"""Performs tree updates, exploration updates, and model optimization."""
        if self.learn.model.training:
            # if self.learn.data.x.items[-1].done: self.learn.current_n_episodes_to_train+=1
            self.learn.memory.update(item=self.learn.data.x.items[-1])
            if len(self.learn.data.x.items)>10:self.learn.data.x.items=np.delete(self.learn.data.x.items,0)
            self.iteration+=1
            self.learn.exploration_method.update(self.episode, max_episodes=self.max_episodes, explore=self.learn.model.training)

        if self.learn.current_n_episodes_to_train>=self.learn.n_episodes_to_train and not self.learn.warming_up:
            samples=list(self.memory.memory)
            assert int(sum([s.done for s in samples]))==self.learn.current_n_episodes_to_train

            _episode_counter=[0]
            q_vals=[]
            def paint_episodes(x,counter):
                if not bool(x.done): return counter[0]
                else:
                    counter[0]+=1
                    return counter[0]
            for g_n,o in groupby([s for s in samples],partial(paint_episodes,counter=_episode_counter)):
                q_vals.extend(calc_qvals([float(s.reward) for s in o],self.learn.discount))

            loss=calc_loss(
                states=torch.cat([s.s for s in samples]),
                actions=torch.cat([s.a for s in samples]),
                logits=self.learn.model(torch.cat([s.s for s in samples])),
                q_vals=torch.Tensor(q_vals).to(self.learn.data.device)
            )
            self.learn.memory.memory.clear()
            self.loss=loss.detach().cpu()
            return {'last_output': loss}
        return {'last_output':self.loss}

    def on_batch_end(self, **kwargs:Any) ->None:
        if self.iteration % 300 == 0:
            self.learn.target_net.sync()


class ReinforceLearner(AgentLearner):
    def __init__(self, data: MDPDataBunch, model, trainers,loss_func=None,episodes_to_train=4,discount=0.99,opt=torch.optim.Adam,**learn_kwargs):
        super().__init__(data=data, model=model, opt=opt,loss_func=loss_func, **learn_kwargs)
        self._loss_func=loss_func
        self.memory=ExperienceReplay(100000)
        self.discount=discount

        self.target_net=TargetNet(self.model)
        self.exploration_method=PolicyExploration(apply_softmax=True)
        self.trainers=listify(trainers)
        for t in self.trainers: self.callbacks.append(t(self))
        self.n_episodes_to_train=episodes_to_train
        self.current_n_episodes_to_train=0
        self.stay_warmed_up_toggle=True

    def init(self, init):pass
    # def init_loss_func(self):pass
    def remove_loss_func(self):
        self.loss_func=None

    @property
    def warming_up(self):
        if self.n_episodes_to_train<=self.current_n_episodes_to_train:
            self.stay_warmed_up_toggle=False
        return self.stay_warmed_up_toggle

    def predict(self, element, **kwargs):
        q_v=self.model(element)
        actions=self.exploration_method.perturb(q_v,self.data.action.action_space)
        return actions


