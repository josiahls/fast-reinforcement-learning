from functools import partial

from fastai.basic_train import Recorder, defaults

from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import fit


class AgentLearner(object):
    silent: bool = None

    def __init__(self, data: MDPDataBunch, model: BaseAgent):
        """
        Will very soon subclass the fastai learner class. For now we need to understand the important functional
        requirements needed in a learner.

        """
        self.model = model
        self.data = data
        self.opt = self.model.opt
        if self.silent is None: self.silent = defaults.silent
        self.add_time: bool = True
        self.callbacks = [partial(Recorder, add_time=self.add_time, silent=self.silent)]
        self.callbacks = [f(learn=self) for f in self.callbacks] + [f(learn=self) for f in self.model.learner_callbacks]

    def predict(self, element):
        return self.model.pick_action(element)

    def fit(self, epochs):
        fit(epochs, self, self.callbacks, [])
