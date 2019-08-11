from fastai.basic_train import LearnerCallback, Any

from fast_rl.agents.BaseAgent import BaseAgent, create_nn_model
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MDPDataBunch
from fast_rl.core.agent_core import ExperienceReplay


class BaseDQNCallback(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.max_episodes = 0
        self.episode = 0

    def on_train_begin(self, max_episodes, **kwargs: Any):
        self.max_episodes = max_episodes

    def on_epoch_begin(self, episode, **kwargs: Any):
        self.episode = episode

    def on_step_end(self, learn: AgentLearner, **kwargs: Any):
        learn.model.memory.update(learn.data.x[-1])
        learn.model.epsilon = learn.model.decay_rate / ()


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
        self.epsilon = 1
        self.decay_rate = 0.1
        self.memory = ExperienceReplay(1000)
        self.action_model = create_nn_model([10, 10, 10], *data.get_action_state_size())

    def forward(self, x):
        x = super(DQN, self).forward(x)
        return self.pick_action(self.action_model(x))
