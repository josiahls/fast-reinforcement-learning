from fastai.basic_train import LearnerCallback

from fast_rl.agents.BaseAgent import BaseAgent, create_nn_model
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch


class BaseDQNCallback(LearnerCallback):
    pass


class DQN(BaseAgent):
    def __init__(self, data: MDPDataBunch):
        """Trains an Agent using the Q Learning method on a neural net.

        Args:
            data:

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).
        """
        super().__init__(data)
        # TODO add recommend cnn based on state size?
        self.model = create_nn_model([10, 10, 10], *data.get_action_state_size())

    def forward(self, x):
        pass