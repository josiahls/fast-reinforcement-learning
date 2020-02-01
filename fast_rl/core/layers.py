r"""`fast_rl.layers` provides essential functions to building and modifying `model` architectures"""
from math import ceil

from fastai.torch_core import *
from fastai.tabular import TabularModel


def init_cnn(mod: Any):
	r""" Utility for initializing cnn Modules. """
	if getattr(mod, 'bias', None) is not None: nn.init.constant_(mod.bias, 0)
	if isinstance(mod, (nn.Conv2d, nn.Linear)): nn.init.kaiming_normal_(mod.weight)
	for sub_mod in mod.children(): init_cnn(sub_mod)


def ks_stride(ks, stride, w, h, n_blocks, kern_proportion=.1, stride_proportion=0.3):
	r""" Utility for determing the the kernel size and stride. """
	kernels, strides, max_dim = [], [], max((w, h))
	for i in range(len(n_blocks)):
		kernels.append(max_dim * kern_proportion)
		strides.append(kernels[-1] * stride_proportion)
		max_dim = (max_dim - kernels[-1]) / strides[-1]
		assert max_dim > 1

	return ifnone(ks, map(ceil, kernels)), ifnone(stride, map(ceil, strides))


class Flatten(nn.Module):
	def forward(self, y): return y.view(y.size(0), -1)


class FakeBatchNorm(Module):
	r""" If we want all the batch norm layers gone, then we will replace the tabular batch norm with this. """
	def forward(self, xi: Tensor, *args): return xi


def conv_bn_lrelu(ni: int, nf: int, ks: int = 3, stride: int = 1, pad=True, bn=True) -> nn.Sequential:
	r""" Create a sequence Conv2d->BatchNorm2d->LeakyReLu layer. (from darknet.py). Allows excluding BatchNorm2d Layer."""
	return nn.Sequential(
		nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=(ks // 2) if pad else 0),
		nn.BatchNorm2d(nf) if bn else FakeBatchNorm(),
		nn.LeakyReLU(negative_slope=0.1, inplace=True))


class ChannelTranspose(Module):
	r""" Runtime image input channel changing. Useful for handling different image channel outputs from different envs. """
	def forward(self, xi: Tensor):
		return xi.transpose(3, 1).transpose(3, 2)


class StateActionSplitter(Module):
	r""" `Actor / Critic` models require breaking the state and action into 2 streams. """

	def forward(self, s_a_tuple: Tuple[Tensor]):
		r""" Returns tensors as -> (State Tensor, Action Tensor) """
		return s_a_tuple[0], s_a_tuple[1]


class StateActionPassThrough(nn.Module):
	r""" Passes action input untouched, but runs the state tensors through a sub module. """
	def __init__(self, layers):
		super().__init__()
		self.layers = layers

	def forward(self, state_action):
		return self.layers(state_action[0]), state_action[1]


class TabularEmbedWrapper(Module):
	r""" Basic `TabularModel` compatibility wrapper. Typically, state inputs will be either categorical or continuous. """
	def __init__(self, tabular_model: TabularModel):
		super().__init__()
		self.tabular_model = tabular_model

	def forward(self, xi: Tensor, *args):
		return self.tabular_model(xi, xi)


class CriticTabularEmbedWrapper(Module):
	r""" Similar to `TabularEmbedWrapper` but assumes input is state / action and requires concatenation. """
	def __init__(self, tabular_model: TabularModel, exclude_cat):
		super().__init__()
		self.tabular_model = tabular_model
		self.exclude_cat = exclude_cat

	def forward(self, args):
		if not self.exclude_cat:
			return self.tabular_model(*args)
		else:
			return self.tabular_model(0, torch.cat(args, axis=1))
