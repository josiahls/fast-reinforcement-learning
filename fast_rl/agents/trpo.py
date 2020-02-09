from fastai.basic_train import LearnerCallback

from fast_rl.core.agent_core import Experience, ExplorationStrategy
from fast_rl.core.basic_train import AgentLearner, listify, torch, Any, List
from fast_rl.core.data_block import MDPDataBunch, MDPStep


class TRPOLearner(AgentLearner):
	def __init__(self,data:MDPDataBunch,model,memory,trainers,opt=torch.optim.RMSprop,
			     **learn_kwargs):
		self.memory:Experience=memory
		super().__init__(data=data,model=model,opt=opt,**learn_kwargs)
		self.trainers=listify(trainers)
		for t in self.trainers: self.callbacks.append(t(self))


class TRPOTrainer(LearnerCallback):
	@property
	def learn(self)->TRPOLearner:
		return self._learn()

	def on_loss_begin(self, **kwargs: Any):
		"""Performs tree updates, exploration updates, and model optimization."""
		if self.learn.model.training: self.learn.memory.update(item=self.learn.data.x.items[-1])
		if not self.learn.warming_up:
			samples: List[MDPStep]=self.memory.sample(self.learn.data.bs)
			self.learn.model.optimize(samples,visitation_freq=1/len(self.learn.memory))
