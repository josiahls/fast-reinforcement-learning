from fastai.basic_train import LearnerCallback
from fastai.tabular.data import emb_sz_rule

from fast_rl.agents.dqn_models import *
from fast_rl.core.agent_core import ExperienceReplay, ExplorationStrategy, Experience
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPDataBunch, FEED_TYPE_STATE, FEED_TYPE_IMAGE, MDPStep


class DQNLearner(AgentLearner):
    def __init__(self, data: MDPDataBunch, model, memory, exploration_method, trainers, opt=torch.optim.RMSprop,
                 **learn_kwargs):
        self.memory: Experience = memory
        self.exploration_method: ExplorationStrategy = exploration_method
        super().__init__(data=data, model=model, opt=opt, **learn_kwargs)
        self.trainers = listify(trainers)
        for t in self.trainers: self.callbacks.append(t(self))

    def predict(self, element, **kwargs):
        training = self.model.training
        if element.shape[0] == 1: self.model.eval()
        pred = self.model(element)
        if training: self.model.train()
        return self.exploration_method.perturb(torch.argmax(pred, axis=1), self.data.action.action_space)

    def interpret_q(self, item):
        with torch.no_grad():
            return torch.sum(self.model(item.s)).cpu().numpy().item()


class FixedTargetDQNTrainer(LearnerCallback):
    def __init__(self, learn, copy_over_frequency=3):
        r"""Handles updating the target model in a fixed target DQN.

        Args:
            learn: Basic Learner.
            copy_over_frequency: For every N iterations we want to update the target model.
        """
        super().__init__(learn)
        self._order = 1
        self.iteration = 0
        self.copy_over_frequency = copy_over_frequency

    def on_step_end(self, **kwargs: Any):
        self.iteration += 1
        if self.iteration % self.copy_over_frequency == 0 and self.learn.model.training:
            self.learn.model.target_copy_over()


class BaseDQNTrainer(LearnerCallback):
    def __init__(self, learn: DQNLearner, max_episodes=None):
        r"""Handles basic DQN end of step model optimization."""
        super().__init__(learn)
        self.n_skipped = 0
        self._persist = max_episodes is not None
        self.max_episodes = max_episodes
        self.episode = -1
        self.iteration = 0
        # For the callback handler
        self._order = 0
        self.previous_item = None

    @property
    def learn(self) -> DQNLearner:
        return self._learn()

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs if not self._persist else self.max_episodes

    def on_epoch_begin(self, epoch, **kwargs: Any):
        self.episode = epoch if not self._persist else self.episode + 1
        self.iteration = 0

    def on_loss_begin(self, **kwargs: Any):
        r"""Performs tree updates, exploration updates, and model optimization."""
        if self.learn.model.training: self.learn.memory.update(item=self.learn.data.x.items[-1])
        self.learn.exploration_method.update(self.episode, max_episodes=self.max_episodes, explore=self.learn.model.training)
        if not self.learn.warming_up:
            samples: List[MDPStep] = self.memory.sample(self.learn.data.bs)
            post_optimize = self.learn.model.optimize(samples)
            if self.learn.model.training: self.learn.memory.refresh(post_optimize=post_optimize)
            self.iteration += 1


def create_dqn_model(data: MDPDataBunch, base_arch: DQNModule, layers=None, ignore_embed=False, channels=None,
                     opt=torch.optim.RMSprop, loss_func=None, lr=0.001, **kwargs):
    bs,state,action=data.bs,data.state,data.action
    nc, w, h, n_conv_blocks = -1, -1, -1, [] if state.mode == FEED_TYPE_STATE else ifnone(channels, [16, 16, 16])
    if state.mode == FEED_TYPE_IMAGE: nc, w, h = state.s.shape[3], state.s.shape[2], state.s.shape[1]
    _layers = ifnone(layers, [64, 64])
    if ignore_embed or np.any(state.n_possible_values == np.inf) or state.mode == FEED_TYPE_IMAGE: emb_szs = []
    else: emb_szs = [(d+1, int(emb_sz_rule(d))) for d in state.n_possible_values.reshape(-1, )]
    ao = int(action.n_possible_values[0])
    model = base_arch(ni=state.s.shape[1], ao=ao, layers=_layers, emb_szs=emb_szs, n_conv_blocks=n_conv_blocks,
                      nc=nc, w=w, h=h, opt=opt, loss_func=loss_func, lr=lr, **kwargs)
    return model


dqn_config = {
    DQNModule: [BaseDQNTrainer],
    DoubleDQNModule: [BaseDQNTrainer, FixedTargetDQNTrainer],
    DuelingDQNModule: [BaseDQNTrainer, FixedTargetDQNTrainer],
    DoubleDuelingModule: [BaseDQNTrainer, FixedTargetDQNTrainer],
    FixedTargetDQNModule: [BaseDQNTrainer, FixedTargetDQNTrainer]
}


def dqn_learner(data: MDPDataBunch, model: DQNModule, memory: ExperienceReplay, exploration_method: ExplorationStrategy,
                trainers=None, copy_over_frequency=300, **kwargs):
    trainers = ifnone(trainers, [c if c != FixedTargetDQNTrainer else partial(c, copy_over_frequency=copy_over_frequency)
                                 for c in dqn_config[model.__class__]])
    return DQNLearner(data, model, memory, exploration_method, trainers, **kwargs)
