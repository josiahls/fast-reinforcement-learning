from collections import deque
from math import ceil

import gym
from fastai.basic_train import *
from fastai.torch_core import *

from fast_rl.core.data_structures import SumTree


class ExplorationStrategy:
	def __init__(self, explore: bool = True): self.explore=explore
	def update(self, max_episodes, explore, **kwargs): self.explore=explore
	def perturb(self, action, action_space) -> np.ndarray:
		"""
		Base method just returns the action. Subclass, and change to return randomly / augmented actions.

		Should use `do_exploration` field. It is recommended that when you subclass / overload, you allow this field
		to completely bypass these actions.

		Args:
			action (np.array): 			Action input as a regular numpy array.
			action_space (gym.Space): 	The original gym space. Should contain information on the action type, and
										possible convenience methods for random action selection.
		"""
		_=action_space
		return action


class GreedyEpsilon(ExplorationStrategy):
	def __init__(self, epsilon_start, epsilon_end, decay, start_episode=0, end_episode=0, **kwargs):
		super().__init__(**kwargs)
		self.end_episode=end_episode
		self.start_episode=start_episode
		self.decay=decay
		self.e_end=epsilon_end
		self.e_start=epsilon_start
		self.epsilon=self.e_start
		self.steps=0

	def perturb(self, action, action_space: gym.Space):
		return action_space.sample() if np.random.random()<self.epsilon and self.explore else action

	def update(self, episode, end_episode=0, **kwargs):
		super(GreedyEpsilon, self).update(**kwargs)
		if self.explore:
			self.end_episode=end_episode
			self.epsilon=self.e_end+(self.e_start-self.e_end)*math.exp(-1.*(self.steps*self.decay))
			self.steps+=1


class OrnsteinUhlenbeck(GreedyEpsilon):
	def __init__(self, size, mu=0., theta=0.15, sigma=0.2, **kwargs):
		"""

		References:
			[1] From https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
			[2] Cumulatively based on

		Args:
			epsilon_start:
			epsilon_end:
			decay:
			**kwargs:
		"""
		super().__init__(**kwargs)
		self.sigma=sigma
		self.theta=theta
		self.mu=mu
		self.x=np.ones(size)

	def perturb(self, action, action_space):
		dx=np.zeros(self.x.shape)
		if self.explore:
			dx=self.theta*(self.mu-self.x)+self.sigma*np.array([np.random.normal() for _ in range(len(self.x))])

		self.x+=dx
		return self.epsilon*self.x+action


class Experience:
	def __init__(self, memory_size, reduce_ram=False):
		self.reduce_ram=reduce_ram
		self.max_size=memory_size
		self.callbacks=[]

	@property
	def memory(self): return None
	def sample(self, **kwargs): pass
	def update(self, item, **kwargs): item.to(device=defaults.device)
	def refresh(self, **kwargs): pass


class ExperienceReplay(Experience):
	def __init__(self, memory_size, **kwargs):
		r"""
		Basic store-er of s space transitions for training agents.

		References:
			[1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
			arXiv preprint arXiv:1312.5602 (2013).

		Args:
			memory_size (int): Max N samples to store
		"""
		super().__init__(memory_size, **kwargs)
		self.max_size=memory_size
		self._memory=deque(maxlen=memory_size)

	@property
	def memory(self):
		return self._memory

	def __len__(self):
		return len(self._memory)

	def sample(self, batch, **kwargs):
		if len(self._memory)<batch: return self._memory
		return random.sample(self.memory, batch)

	def update(self, item, **kwargs):
		item=deepcopy(item)
		super().update(item, **kwargs)
		if self.reduce_ram: item.clean()
		self._memory.append(item)


class PriorityExperienceReplayCallback(LearnerCallback):
	def on_train_begin(self, **kwargs):
		self.learn.model.loss_func=partial(self.learn.model.memory.handle_loss, loss_fn=self.learn.model.loss_func)


class PriorityExperienceReplay(Experience):

	def handle_loss(self, y, y_hat, loss_fn):
		return (loss_fn(y, y_hat)*torch.from_numpy(self.p_weights).to(device=defaults.device).float()).mean()

	def __init__(self, memory_size, batch_size=64, epsilon=0.01, alpha=0.6, beta=0.4, b_inc=-0.001, **kwargs):
		"""
		Prioritizes sampling based on samples requiring the most learning.

		References:
			[1] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

		Args:
			batch_size (int): Size of sample, and thus size of expected index update.
			alpha (float): Changes the sampling behavior 1 (non-uniform) -> 0 (uniform)
			epsilon (float): Keeps the probabilities of items from being 0
			memory_size (int): Max N samples to store
		"""
		super().__init__(memory_size, **kwargs)
		self.batch_size=batch_size
		self.alpha=alpha
		self.beta=beta
		self.b_inc=b_inc
		self.p_weights=None  # np.zeros(self.batch_size, dtype=float)
		self.epsilon=epsilon
		self.tree=SumTree(self.max_size)
		self.callbacks=[PriorityExperienceReplayCallback]
		# When sampled, store the sample indices for refresh.
		self._indices=None  # np.zeros(self.batch_size, dtype=int)

	@property
	def memory(self):
		return self.tree.data

	def __len__(self):
		return self.tree.n_entries

	def refresh(self, post_optimize, **kwargs):
		if post_optimize is not None:
			self.tree.update(self._indices.astype(int), np.abs(post_optimize['td_error'])+self.epsilon)

	def sample(self, batch, **kwargs):
		self.beta=np.min([1., self.beta+self.b_inc])
		ranges=np.linspace(0, ceil(self.tree.total()/batch), num=batch+1)
		uniform_ranges=[np.random.uniform(ranges[i], ranges[i+1]) for i in range(len(ranges)-1)]
		try:
			self._indices, weights, samples=self.tree.batch_get(uniform_ranges)
		except ValueError:
			warn('Too few values to unpack. Your batch size is too small, when PER queries tree, all 0 values get'
				 ' ignored. We will retry until we can return at least one sample.')
			samples=self.sample(batch)
			return samples

		self.p_weights=self.tree.anneal_weights(weights, self.beta)
		return samples

	def update(self, item, **kwargs):
		"""
		Updates the tree of PER.

		Assigns maximal priority per [1] Alg:1, thus guaranteeing that sample being visited once.

		Args:
			item:

		Returns:

		"""
		item=deepcopy(item)
		super().update(item, **kwargs)
		maximal_priority=self.alpha
		if self.reduce_ram: item.clean()
		self.tree.add(np.abs(maximal_priority)+self.epsilon, item)


# class HindsightExperienceReplay(Experience):
# 	def __init__(self, memory_size):
# 		"""
#
# 		References:
# 			[1] Andrychowicz, Marcin, et al. "Hindsight experience replay."
# 			Advances in Neural Information Processing Systems. 2017.
#
# 		Args:
# 			memory_size:
# 		"""
# 		super().__init__(memory_size)
