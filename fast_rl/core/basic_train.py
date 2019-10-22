from functools import partial

from fastai.basic_train import Recorder, defaults, Learner, listify, ifnone

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.core.MarkovDecisionProcess import MDPDataBunchAlpha, MDPMemoryManagerAlpha, MDPCallback
from fast_rl.core.agent_core import fit
from fast_rl.core.metrics import EpsilonMetric


class WrapperLossFunc(object):
    def __init__(self, learn):
        self.learn = learn

    def __call__(self, *args, **kwargs):
        return self.learn.model.loss


class AgentLearner(Learner):

    def __post_init__(self) -> None:
        super().__post_init__()
        self._loss_func = WrapperLossFunc(self)
        self.loss_func = None
        self.callback_fns += self.model.learner_callbacks + [MDPCallback]

    def predict(self, element, **kwargs):
        return self.model.pick_action(element)

    def init_loss_func(self):
        r"""
         Initializes the loss function wrapper for logging loss.

         Since most RL models have a period of warming up such as filling memory buffers, we cannot log any loss.
         By default, the learner will have a `None` loss function, and so the fit function will not try to log that
         loss.
         """
        self.loss_func = self._loss_func


class AgentLearnerAlpha(object):
    silent: bool = None

    def __init__(self, data: MDPDataBunchAlpha, model: BaseAgent, metrics=None):
        """
        Will very soon subclass the fastai learner class. For now we need to understand the important functional
        requirements needed in a learner.

        """
        self.model = model
        self.data = data
        self._loss_func = WrapperLossFunc(self)
        self.loss_func = None
        self.metrics = []
        self.opt = self.model.opt
        if self.silent is None: self.silent = defaults.silent
        self.add_time: bool = True
        self.recorder = Recorder(learn=self, add_time=self.add_time, silent=self.silent)
        self.recorder.no_val = self.data.empty_val
        self.callbacks = ifnone(metrics, [])
        # self.callbacks += [partial(MDPMemoryManagerAlpha, mem_strategy=mem_strategy, k=k)]
        self.callbacks += [MDPCallback]
        self.callbacks = [self.recorder] + [f(self) for f in self.callbacks] + [f(learn=self) for f in self.model.learner_callbacks]

    def init_loss_func(self):
        self.loss_func = self._loss_func

    def predict(self, element):
        return self.model.pick_action(element)

    def fit(self, epochs):
        fit(epochs, self, self.callbacks, metrics=self.metrics)
