from fastai.callback import OptimWrapper

from fast_rl.core.layers import *


class DQNModule(Module):

	def __init__(self, ni: int, ao: int, layers: Collection[int], discount: float = 0.99, lr=0.001,
				n_conv_blocks: Collection[int] = 0, nc=3, opt=None, emb_szs: ListSizes = None, loss_func=None,
				w=-1, h=-1, ks: Union[None, list]=None, stride: Union[None, list]=None, grad_clip=5,
				conv_kern_proportion=0.1, stride_proportion=0.1, pad=False, batch_norm=False):
		r"""
		Basic DQN Module.

		Args:
			ni: Number of inputs. Expecting a flat state `[1 x ni]`
			ao: Number of actions to output.
			layers: Number of layers where is determined per element.
			n_conv_blocks: If `n_conv_blocks` is not 0, then convolutional blocks will be added
						   to the head on top of existing linear layers.
			nc: Number of channels that will be expected by the convolutional blocks.
		"""
		super().__init__()
		self.name = 'DQN'
		self.loss = None
		self.loss_func = loss_func
		self.discount = discount
		self.gradient_clipping_norm = grad_clip
		self.lr = lr
		self.batch_norm = batch_norm
		self.switched = False
		# self.ks, self.stride = ([], []) if len(n_conv_blocks) == 0 else ks_stride(ks, stride, w, h, n_conv_blocks, conv_kern_proportion, stride_proportion)
		self.ks, self.stride=([], []) if len(n_conv_blocks)==0 else (ifnone(ks, [10, 10, 10]), ifnone(stride, [5, 5, 5]))
		self.action_model = nn.Sequential()
		_layers = [conv_bn_lrelu(ch, self.nf, ks=ks, stride=stride, pad=pad, bn=self.batch_norm) for ch, self.nf, ks, stride in zip([nc]+n_conv_blocks[:-1],n_conv_blocks, self.ks, self.stride)]

		if _layers: ni = self.setup_conv_block(_layers=_layers, ni=ni, nc=nc, w=w, h=h)
		self.setup_linear_block(_layers=_layers, ni=ni, nc=nc, w=w, h=h, emb_szs=emb_szs, layers=layers, ao=ao)
		self.init_weights(self.action_model)
		self.opt = None
		self.set_opt(opt)

	def set_opt(self, opt):
		self.opt=OptimWrapper.create(ifnone(optim.Adam, opt), lr=self.lr, layer_groups=[self.action_model])

	def setup_conv_block(self, _layers, ni, nc, w, h):
		self.action_model.add_module('conv_block', nn.Sequential(*(self.fix_switched_channels(ni, nc, _layers) + [Flatten()])))
		training = self.action_model.training
		self.action_model.eval()
		ni = int(self.action_model(torch.zeros((1, w, h, nc) if self.switched else (1, nc, w, h))).view(-1, ).shape[0])
		self.action_model.train(training)
		return ni

	def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao):
		tabular_model = TabularModel(emb_szs=emb_szs, n_cont=ni if not emb_szs else 0, layers=layers, out_sz=ao, use_bn=self.batch_norm)
		if not emb_szs: tabular_model.embeds = None
		if not self.batch_norm: tabular_model.bn_cont = FakeBatchNorm()
		self.action_model.add_module('lin_block', TabularEmbedWrapper(tabular_model))

	def fix_switched_channels(self, current_channels, expected_channels, layers: list):
		if current_channels == expected_channels:
			return layers
		else:
			self.switched = True
			return [ChannelTranspose()] + layers

	def forward(self, xi: Tensor):
		training = self.training
		if xi.shape[0] == 1: self.eval()
		pred = self.action_model(xi)
		if training: self.train()
		return pred

	def init_weights(self, m):
		if type(m) == nn.Linear:
			torch.nn.init.xavier_uniform_(m.weight)
			m.bias.data.fill_(0.01)

	def sample_mask(self, d):
		return torch.sub(1.0, d)

	def optimize(self, sampled):
		r"""Uses ER to optimize the Q-net (without fixed targets).

		Uses the equation:

		.. math::
				Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
				\;|\; s, a \Big]


		Returns (dict): Optimization information

		"""
		with torch.no_grad():
			r = torch.cat([item.reward.float() for item in sampled])
			s_prime = torch.cat([item.s_prime for item in sampled])
			s = torch.cat([item.s for item in sampled])
			a = torch.cat([item.a.long() for item in sampled])
			d = torch.cat([item.done.float() for item in sampled])
		masking = self.sample_mask(d)

		y_hat = self.y_hat(s, a)
		y = self.y(s_prime, masking, r, y_hat)

		loss = self.loss_func(y, y_hat)

		if self.training:
			self.opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), self.gradient_clipping_norm)
			for param in self.action_model.parameters():
				if param.grad is not None: param.grad.data.clamp_(-1, 1)
			self.opt.step()

		with torch.no_grad():
			self.loss = loss
			post_info = {'td_error': to_detach(y - y_hat).cpu().numpy()}
			return post_info

	def y_hat(self, s, a):
		return self.action_model(s).gather(1, a)

	def y(self, s_prime, masking, r, y_hat):
		return self.discount * self.action_model(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)


class FixedTargetDQNModule(DQNModule):
	def __init__(self, ni: int, ao: int, layers: Collection[int], tau=1, **kwargs):
		super().__init__(ni, ao, layers, **kwargs)
		self.name = 'Fixed Target DQN'
		self.tau = tau
		self.target_model = copy(self.action_model)

	def target_copy_over(self):
		r""" Updates the target network from calls in the FixedTargetDQNTrainer callback."""
		# self.target_net.load_state_dict(self.action_model.state_dict())
		for target_param, local_param in zip(self.target_model.parameters(), self.action_model.parameters()):
			target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

	def y(self, s_prime, masking, r, y_hat):
		r"""
		Uses the equation:

		.. math::

				Q^{*}(s, a) = \mathbb{E}_{s'∼ \Big\epsilon} \Big[r + \lambda \displaystyle\max_{a'}(Q^{*}(s' , a'))
				\;|\; s, a \Big]

		"""
		return self.discount * self.target_model(s_prime).max(1)[0].unsqueeze(1) * masking + r.expand_as(y_hat)


class DoubleDQNModule(FixedTargetDQNModule):
	def __init__(self, ni: int, ao: int, layers: Collection[int], **kwargs):
		super().__init__(ni, ao, layers, **kwargs)
		self.name = 'DDQN'

	def calc_y(self, s_prime, masking, r, y_hat):
		return self.discount * self.target_model(s_prime).gather(1, self.action_model(s_prime).argmax(1).unsqueeze(
			1)) * masking + r.expand_as(y_hat)


class DuelingBlock(nn.Module):
	def __init__(self, ao, stream_input_size):
		super().__init__()

		self.val = nn.Linear(stream_input_size, 1)
		self.adv = nn.Linear(stream_input_size, ao)

	def forward(self, xi):
		r"""Splits the base neural net output into 2 streams to evaluate the advantage and v of the s space and
		corresponding actions.

		.. math::
		   Q(s,a;\; \Theta, \\alpha, \\beta) = V(s;\; \Theta, \\beta) + A(s, a;\; \Theta, \\alpha) - \\frac{1}{|A|}
		   \\Big\\sum_{a'} A(s, a';\; \Theta, \\alpha)

		"""
		val, adv = self.val(xi), self.adv(xi)
		xi = val.expand_as(adv) + (adv - adv.mean()).squeeze(0)
		return xi


class DuelingDQNModule(FixedTargetDQNModule):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.name = 'Dueling DQN'

	def setup_linear_block(self, _layers, ni, nc, w, h, emb_szs, layers, ao):
		tabular_model = TabularModel(emb_szs=emb_szs, n_cont=ni if not emb_szs else 0, layers=layers, out_sz=ao,
							 use_bn=self.batch_norm)
		if not emb_szs: tabular_model.embeds = None
		if not self.batch_norm: tabular_model.bn_cont = FakeBatchNorm()
		tabular_model.layers, removed_layer = split_model(tabular_model.layers, [last_layer(tabular_model)])
		ni = removed_layer[0].in_features
		self.action_model.add_module('lin_block', TabularEmbedWrapper(tabular_model))
		self.action_model.add_module('dueling_block', DuelingBlock(ao, ni))


class DoubleDuelingModule(DuelingDQNModule, DoubleDQNModule):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.name = 'DDDQN'
