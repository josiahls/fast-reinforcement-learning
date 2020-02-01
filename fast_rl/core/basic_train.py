from multiprocessing.pool import Pool

from fastai.basic_train import Learner, load_callback
from fastai.torch_core import *

from fast_rl.core.data_block import MDPDataBunch


class WrapperLossFunc(object):
	def __init__(self, learn):
		self.learn = learn

	def __call__(self, *args, **kwargs):
		return self.learn.model.loss



def load_learner(path:PathOrStr, file:PathLikeOrBinaryStream='export.pkl', **db_kwargs):
	r""" Similar to fastai `load_learner`, handles load_state for data differently. """
	source = Path(path)/file if is_pathlike(file) else file
	state = torch.load(source, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(source)
	model = state.pop('model')
	data = MDPDataBunch.load_state(path, state.pop('data'))
	# if test is not None: src.add_test(test)
	# data = src.databunch(**db_kwargs)
	cb_state = state.pop('cb_state')
	clas_func = state.pop('cls')
	res = clas_func(data, model, **state)
	res.callback_fns = state['callback_fns'] #to avoid duplicates
	res.callbacks = [load_callback(c,s, res) for c,s in cb_state.items()]
	return res


class AgentLearner(Learner):

	def __init__(self, data, loss_func=None, callback_fns=None, opt=torch.optim.Adam, **kwargs):
		super().__init__(data=data, callback_fns=ifnone(callback_fns, []) + data.callback, **kwargs)
		self.model.loss_func = ifnone(loss_func, F.mse_loss)
		self.model.set_opt(opt)
		self.loss_func = None
		self.trainers = None
		self._loss_func = WrapperLossFunc(self)

	@property
	def warming_up(self):
		return self.data.bs > len(self.data.x)

	def init_loss_func(self):
		r"""
		 Initializes the loss function wrapper for logging loss.

		 Since most RL models have a period of warming up such as filling tree buffers, we cannot log any loss.
		 By default, the learner will have a `None` loss function, and so the fit function will not try to log that
		 loss.
		 """
		self.loss_func = WrapperLossFunc(self)

	def export(self, file:PathLikeOrBinaryStream='export.pkl', destroy=False, pickle_data=False):
		"Export the state of the `Learner` in `self.path/file`. `file` can be file-like (file or buffer)"
		if rank_distrib(): return # don't save if slave proc
		# For now we exclude the 'loss_func' since it is pointing to a model loss.
		args = ['opt_func', 'metrics', 'true_wd', 'bn_wd', 'wd', 'train_bn', 'model_dir', 'callback_fns', 'memory',
				'exploration_method', 'trainers']
		state = {a:getattr(self,a) for a in args}
		state['cb_state'] = {cb.__class__:cb.get_state() for cb in self.callbacks}
		#layer_groups -> need to find a way
		#TO SEE: do we save model structure and weights separately?
		with ModelOnCPU(self.model) as m:
			m.opt = None
			state['model'] = m
			xtra = dict(normalize=self.data.norm.keywords) if getattr(self.data, 'norm', False) else {}
			state['data'] = self.data.train_ds.get_state(**xtra) if self.data.valid_dl is None else self.data.valid_ds.get_state(**xtra)
			state['data']['add_valid'] = not self.data.empty_val
			if pickle_data: self.data.to_pickle(self.data.path)
			state['cls'] = self.__class__
			try_save(state, self.path, file)
		if destroy: self.destroy()

	def interpret_q(self, xi):
		raise NotImplemented


class PipeLine(object):
	def __init__(self, n_threads, pipe_line_function):
		warn(Warning('Currently not super useful. Seems to have issues with running a single env in multiple threads.'))
		self.pipe_line_function = pipe_line_function
		self.n_threads = n_threads
		self.pool = Pool(self.n_threads)

	def start(self, n_runs):
		return self.pool.map(self.pipe_line_function,  range(n_runs))