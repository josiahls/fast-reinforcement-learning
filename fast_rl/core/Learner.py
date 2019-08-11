from fast_rl.agents.BaseAgent import BaseAgent
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


class AgentLearner(object):

    def __init__(self, data: MDPDataBunch, model: BaseAgent):
        """
        Will very soon subclass the fastai learner class. For now we need to understand the important functional
        requirements needed in a learner.

        """
        self.model = model
        self.data = data

    def predict(self, element):
        return self.model(element)
