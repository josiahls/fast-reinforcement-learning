from typing import *
from fastai.basic_train import LearnerCallback, Module, torch, ifnone, Tensor, listify

import numpy as np
from fastai.tabular import TabularModel
from torch.distributions import Categorical
from torch.nn import Sequential

from fast_rl.agents.reinforce import lazy_conv_out
from fast_rl.core.agent_core import ExplorationStrategy
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPStep
from fast_rl.core.layers import conv_bn_lrelu, ChannelTranspose, Flatten, FakeBatchNorm, TabularEmbedWrapper



class REINFORCEModel(Module):
	def __init__(self, ni: int, na: int, layers: Optional[List[int]] = None, conv_layers: Optional[List[int]] = None,
			stride: Optional[List[int]] = None, padding: Optional[List[int]] = None, use_bn=False,
			nc: Optional[int] = None,
			w: Optional[int] = None, h: Optional[int] = None, embed_szs: Optional[List[int]] = None):
		super().__init__()
		self.switched=False
		self.action_model=Sequential()
		if self.setup_convolutional_layers(ni, nc, conv_layers, stride, padding, use_bn):
			ni=lazy_conv_out(self.action_model, w, h, nc, self.switched)
		self.setup_linear_layers(ni, ifnone(embed_szs, []), ifnone(layers, [32, 32]), na, use_bn)

	def set_opt(self, _):
		pass

	def fix_switched_channels(self, current_channels, expected_channels, layers: list):
		if current_channels==expected_channels: return layers
		self.switched=True
		return [ChannelTranspose()]+layers

	def setup_convolutional_layers(self, ni, nc, cv_l, stride, padding, use_bn) -> bool:
		if cv_l is None or len(cv_l)==0: return False
		# gen a list of conv blocks based on the input size and the list of filter sizes
		conv_blocks=[conv_bn_lrelu(_ni, nf, s, p, bn=use_bn) for _ni, nf, s, p in
					 zip([ni]+cv_l[:-1], cv_l[1:], stride, padding)]
		fixed_conv_blocks=self.fix_switched_channels(ni, nc, conv_blocks)
		self.action_model.add_module('conv_block', Sequential(fixed_conv_blocks+[Flatten()]))
		return True

	def setup_linear_layers(self, ni, emb_szs, layers, ao, use_bn):
		tabular_model=TabularModel(emb_szs=emb_szs, n_cont=ni if not emb_szs else 0, layers=layers, out_sz=ao,
			use_bn=use_bn)
		if not emb_szs: tabular_model.embeds=None
		if not use_bn: tabular_model.bn_cont=FakeBatchNorm()
		self.action_model.add_module('lin_block', TabularEmbedWrapper(tabular_model))

	def forward(self, xi: Tensor):
		training=self.training
		if xi.shape[0]==1: self.eval()
		pred=self.action_model(xi)
		if training: self.train()
		return pred