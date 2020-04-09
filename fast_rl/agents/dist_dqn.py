import collections
from copy import deepcopy

from fastai.basic_train import LearnerCallback
from fastai.imports import torch, Any

from fast_rl.agents.dist_dqn_models import TargetNet
from fast_rl.core.agent_core import ExperienceReplay, NStepExperienceReplay
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

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state','done'))




def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return proj_distr



def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.done)
        # if exp.last_state is None:
        #     last_states.append(state)       # the result will be masked anyway
        # else:
        last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss(batch, net, tgt_net, gamma, device="cpu", save_prefix=None):
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    distr_v = net(states_v)
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()



class BaseDistDQNTrainer(LearnerCallback):
    def __init__(self, learn: 'DistDQNLearner', max_episodes=None):
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

    @property
    def learn(self)->'DistDQNLearner':
        return self._learn()

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs if not self._persist else self.max_episodes

    def on_epoch_begin(self, epoch, **kwargs: Any):
        pass

    def on_backward_begin(self, **kwargs: Any):return {'skip_bwd': self.learn.warming_up}
    def on_backward_end(self, **kwargs:Any): return {'skip_step':False}
    def on_step_end(self, **kwargs: Any):return {'skip_zero': False}

    def on_loss_begin(self, **kwargs: Any):
        r"""Performs tree updates, exploration updates, and model optimization."""
        if self.learn.model.training:
            self.learn.memory.update(item=self.learn.data.x.items[-1])
            self.iteration+=1
            self.learn.epsilon_tracker.frame(self.iteration)

        if not self.learn.warming_up:
            samples: List[MDPStep]=self.memory.sample(self.learn.data.bs)
            batch=[ExperienceFirstLast(state=deepcopy(s.s[0]),action=deepcopy(s.action.taken_action),
                   reward=deepcopy(s.reward),last_state=deepcopy(s.s_prime[0]),done=deepcopy(s.done)) for s in samples]
            # model_func=lambda x: self.learn.model.qvals(x)
            loss=calc_loss(batch,self.learn.model,self.learn.target_net.target_model,gamma=0.99,device=self.learn.data.device,save_prefix=None)
            return {'last_output':loss}
        else: return None

    def on_batch_end(self, **kwargs:Any) ->None:
        if self.iteration % 300 == 0:
            self.learn.target_net.sync()


class ArgmaxActionSelector(object):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(object):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, sum(mask))
        actions[mask] = rand_actions
        return actions

class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)



class DistDQNLearner(AgentLearner):
    def __init__(self, data: MDPDataBunch, model, trainers, loss_func=None,opt=torch.optim.Adam,**learn_kwargs):
        super().__init__(data=data, model=model, opt=opt,loss_func=loss_func, **learn_kwargs)
        self._loss_func=loss_func
        self.memory=NStepExperienceReplay(100000)
        self.target_net=TargetNet(self.model)
        self.exploration_method=EpsilonGreedyActionSelector(1.0)
        self.epsilon_tracker=EpsilonTracker(self.exploration_method, {'epsilon_frames': 100, 'epsilon_start': 1.0,
                                                                     'epsilon_final': 0.02})
        self.trainers=listify(trainers)
        for t in self.trainers: self.callbacks.append(t(self))

    def init(self, init):pass
    # def init_loss_func(self):pass

    def predict(self, element, **kwargs):
        model_func=lambda x: self.model.qvals(x)
        q_v=model_func(element)
        q=q_v.data.cpu().numpy()
        actions=self.exploration_method(q)
        return actions

