from typing import *

import gym
import numpy as np
from fastai.basic_train import LearnerCallback, torch, ifnone, listify, OptimWrapper
from torch import nn

from fast_rl.core.agent_core import ExplorationStrategy, Experience
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPStep


class EpisodeBuffer(Experience):
	def __init__(self, memory_size,**kwargs):
		super().__init__(memory_size,**kwargs)
		self.episodes:List[Dict[str,List[MDPStep]]]=[{}]
		self.current_episode_reward=0

	def __len__(self): return len(self.episodes)

	def not_empty_episodes(self): return [e for e in self.episodes if e]

	def update(self, item, **kwargs):
		if 'episode' not in self.episodes[-1]: self.episodes[-1]['episode']=[]
		self.episodes[-1]['episode'].append(item)
		self.current_episode_reward+=item.reward.item()
		if item.d:
			self.episodes[-1]['reward']=self.current_episode_reward
			self.current_episode_reward=0
			self.episodes.append({})


class CEMTrainer(LearnerCallback):
	def __init__(self, learn):
		super().__init__(learn)
		self.cache_loss=None

	def on_batch_begin(self,**kwargs):
		return {'last_target':self.learn.data.x.items[-1].a.squeeze(0).to(device=self.learn.data.device)}

	def on_backward_begin(self,smooth_loss, **kwargs:Any):
		self.cache_loss=ifnone(self.cache_loss,smooth_loss)
		if len(self.learn.memory)<self.learn.data.bs: return {'skip_bwd':True,'skip_step':True,'skip_zero':True,
															  'last_loss':self.cache_loss,'smooth_loss':self.cache_loss}
		self.cache_loss=self.learn.optimize()
		return {'skip_bwd':False,'last_loss':self.cache_loss,'skip_step':False,'skip_zero':False,'smooth_loss':self.cache_loss}

	def on_backward_end(self, **kwargs: Any):
		if len(self.learn.memory)<self.learn.data.bs: return {'skip_step': True}
		return {'skip_step': False}

	def on_step_end(self, **kwargs: Any):
		if len(self.learn.memory)<self.learn.data.bs: return {'skip_zero': True}
		self.memory.episodes=[{}]
		return {'skip_zero': False}

	def on_loss_begin(self, **kwargs:Any) ->None:
		if self.learn.model.training: self.learn.memory.update(item=self.learn.data.x.items[-1])


class Probabilistic(ExplorationStrategy):
	def __init__(self):
		super().__init__()
		self.sm=nn.Softmax(dim=1)

	def perturb(self, action, action_space: gym.Space):
		action=self.sm(action)
		a_prob=action.squeeze(0).data.detach().cpu().numpy()
		return np.random.choice(len(a_prob),p=a_prob)


class CEMLearner(AgentLearner):
	def __init__(self, data,model,percentile=70,trainers=None,lr=0.01,exploration_strategy=None,wd=0,**kwargs):
		self.percentile=percentile
		trainers=ifnone(trainers,CEMTrainer)
		super().__init__(data=data, model=model,wd=wd,**kwargs)
		self.opt=OptimWrapper.create(self.opt_func, lr=lr,layer_groups=[self.model.action_model])
		self.loss_func=nn.CrossEntropyLoss()
		self.exploration_strategy=ifnone(exploration_strategy,Probabilistic())
		self.trainers=listify(trainers)
		self.memory=EpisodeBuffer(self.data.batch_size)
		for t in self.trainers: self.callbacks.append(t(self))

	def filter_memory(self):
		episodes=self.memory.not_empty_episodes()
		r=list(map(lambda x: x['reward'],episodes))
		r_boundary=np.percentile(r,self.percentile)
		r_mean=float(np.mean(r))

		s=[]
		a=[]
		for e in episodes:
			if e['reward']<r_boundary: continue
			s.extend(map(lambda x: x.s, e['episode']))
			a.extend(map(lambda x: x.a, e['episode']))
		return s,a,r_boundary,r_mean

	def optimize(self):
		self.opt.zero_grad()
		s,a,boundary,r_mean=self.filter_memory()
		s,a=torch.cat(s).to(device=self.data.device),torch.cat(a).to(device=self.data.device).squeeze(1)
		a_scores=self.model(s)
		loss=self.loss_func(a_scores, a)
		return loss

	def predict(self, element, **kwargs):
		training=self.model.training
		if element.shape[0]==1: self.model.eval()
		pred=self.model(element)
		if training: self.model.train()
		return self.exploration_strategy.perturb(pred, self.data.action.action_space)