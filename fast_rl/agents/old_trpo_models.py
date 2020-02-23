from fastai.torch_core import *
from scipy.stats import entropy


class QModule(Module):
	def __init__(self, ni, na):
		super().__init__()
		self.ni,self.na=ni,na

class VModule(Module):
	def __init__(self, ni):
		super().__init__()
		self.ni=ni


def kl_divergence(p: np.array, q: np.array): return entropy(p, q)
def advantage(s,a,q_net:QModule,v_net:VModule): return q_net(s,a)-v_net(s)

def update_q(): pass
def update_v(): pass


class TRPOModule(Module):
	"""
	Implementation of the TRPO (Trust Region Policy Optimization) algorithm.

	 Policy Gradient based algorithm for reinforcement learning in discrete
	 and continuous state and action spaces. Details of the algorithm's mathematical background can be found in [1].

	References:
		[1] (Schulman et al., 2017) Trust Region Policy Optimization.
	"""
	def __init__(self,ni:int,na:int,discount:float,fc_layers:List[int]=None,conv_filters:List[int]=None,nc=3,bn=False,
				 q_lr=1e-3,v_lr=1e-4,ks:List[int]=None,stride:List[int]=None,discrete:bool=True,paths='single',
				 n_timesteps=5,training=True,kl_e=0.1):
		r"""
		Implementation of the TRPO (Trust Region Policy Optimization) algorithm.

		Args:
			ni: Number of inputs. Typically will be the number of dimensions of the state space.
			na: Number of actions. Typically the number of action dimensions.
			discount: Discount factor of q value estimation.
			fc_layers: If the state input is not image based, then this is required.
			conv_filters: Alternative image based state input/
			nc: Number of channels of input images
			bn: Whether to use batch norm. If True typically produces unfavorable results, so default is False.
			q_lr: Learning rate for the q value neural net.
			v_lr: Learning rate for the v value neural net.
			ks: Kernel size for conv layers for image based state inputs.
			stride: Stride for conv layers for image based state inputs.
			discrete: Whether the action space is discrete or not.
			paths: TRPO has 2 options: vine or single paths.
			n_timesteps:
		"""
		super().__init__()
		self.kl_e=kl_e
		self.discount=discount
		self.discrete=discrete
		self.q_net=QModule(ni,na)
		self.v_net=VModule(ni)
		self.training=True

	def forward(self, xi):
		training = self.training
		if xi.shape[0] == 1: self.eval()
		pred = self.v_net(xi)
		if training: self.train()
		return pred

	def step(self,samples:List,visitation_freq:float):
		# s_0 ~ p_0(s_0)
		with torch.no_grad():
			r = torch.cat([item.reward.float() for item in samples])
			s_prime = torch.cat([item.s_prime for item in samples])
			s = torch.cat([item.s for item in samples])
			# [1] actions are chosen according to pi
			a=self.forward(s)

		# p_pi(s) = P(s_0=s)+discount*P(s_1=s)+discount^2*P(s_2=s) ...
		p=sum([visitation_freq*(self.discount**i) for i in range(len(samples))])
		# policy_prime(s) = argmax_a (A_pi (s,a))
		policy_prime=torch.argmax(advantage(s,a,self.q_net,self.v_net),1)

		adv=advantage(s,a,self.q_net,self.v_net)

		# C = (4*e*discount) / (1 - discount) ^ 2
		c=(4*self.kl_e*self.discount)/((1-self.discount)**2)

		v_net_target=deepcopy(self.v_net)
		for target_param, local_param in zip(v_net_target.parameters(), self.v_net.parameters()):
			# n(pi)= sum_s(p_pi(s) * sum_a (policy_prime(s,a)) * A(s,a))
			l_pi=target_param.data+p*sum(policy_prime*adv)
			# pi_prime=L_pi(policy) - C * D(pi_i+1,pi)
			target_param.data.copy_(l_pi-c*kl_divergence(target_param.data,local_param.data))

