from multiprocessing.pool import Pool

from fastai.basic_train import Learner, warn, ifnone, F, List


class WrapperLossFunc(object):
    def __init__(self, learn):
        self.learn = learn

    def __call__(self, *args, **kwargs):
        return self.learn.model.loss


class AgentLearner(Learner):

    def __init__(self, data, loss_func=None, callback_fns=None, **kwargs):
        super().__init__(data=data, callback_fns=ifnone(callback_fns, []) + data.callback, **kwargs)
        self.model.loss_func = ifnone(loss_func, F.mse_loss)
        self.loss_func = None
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
        self.loss_func = self._loss_func

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