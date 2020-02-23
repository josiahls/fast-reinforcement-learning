from typing import *
from fastai.basic_train import LearnerCallback, Module, torch, ifnone, Tensor, listify

import numpy as np
from fastai.tabular import TabularModel
from torch.distributions import Categorical
from torch.nn import Sequential

from fast_rl.core.agent_core import ExplorationStrategy
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPStep
from fast_rl.core.layers import conv_bn_lrelu, ChannelTranspose, Flatten, FakeBatchNorm, TabularEmbedWrapper


class GaussianBasedExploration(ExplorationStrategy):
	r""" Exploration via gaussian distribution of action outputs.

	This is per the usefulness noted in [1] pg 15.

	References:
		[1] .. (Williams, 1992) [REINFORCE] Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
	"""
	def perturb(self, action, action_space) -> np.ndarray:
		m=Categorical(action)
		a=m.sample()
		return a


class LogBasedExploration(ExplorationStrategy):
	def __init__(self):
		super().__init__()
		self.log_prob_a=0

	r""" Exploration via log based probability distribution of action outputs. """
	def perturb(self, action, action_space) -> np.ndarray:
		m=Categorical(action)
		a=m.sample()
		self.log_prob_a=m.log_prob(a)
		return a


class REINFORCEStepWiseTrainer(LearnerCallback):
	def __init__(self, learn):
		super().__init__(learn)
		self._order=1

	def on_loss_begin(self, last_output,**kwargs):
		r""" Loss will require the reward also """
		return {'last_output':{'last_output':last_output,
							   'reward':self.learn.data.x.items[-1].reward.float().to(device=self.learn.data.device),
							   'log_prob':self.learn.exploration_strategy.log_prob_a}}

	def on_step_end(self, **kwargs:Any) ->None:
		self.learn.exploration_strategy.log_prob_a


def log_wise_loss(out, *args):
	return -1*out['log_prob']*out['reward']


class REINFORCELearner(AgentLearner):
	def __init__(self, data,model,trainers=None,**kwargs):
		trainers=ifnone(trainers,REINFORCEStepWiseTrainer)
		super().__init__(data=data, model=model,**kwargs)
		self.loss_func=log_wise_loss
		self.exploration_strategy=LogBasedExploration()
		self.trainers=listify(trainers)
		for t in self.trainers: self.callbacks.append(t(self))

	def predict(self, element, **kwargs):
		training=self.model.training
		if element.shape[0]==1: self.model.eval()
		pred=self.model(element)
		if training: self.model.train()
		return self.exploration_strategy.perturb(pred, self.data.action.action_space)


def lazy_conv_out(m:Module,w,h,nc,switched)->int:
	r""" A Lazier way to determining the conv block output. """
	is_training=m.training
	m.eval()
	ni=int(m(torch.zeros((1, w, h, nc) if switched else (1, nc, w, h))).view(-1, ).shape[0])
	if is_training: m.train(True)
	return ni




