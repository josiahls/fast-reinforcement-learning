from fastai.basic_train import LearnerCallback
import fastai.tabular.data
from fastai.torch_core import *

from fast_rl.agents.ddpg_models import DDPGModule
from fast_rl.core.agent_core import ExperienceReplay, ExplorationStrategy, Experience
from fast_rl.core.basic_train import AgentLearner
from fast_rl.core.data_block import MDPDataBunch, MDPStep, FEED_TYPE_STATE, FEED_TYPE_IMAGE


class DDPGLearner(AgentLearner):
    def __init__(self, data: MDPDataBunch, model, memory, exploration_method, trainers, opt=optim.Adam,
                 **kwargs):
        self.memory: Experience = memory
        self.exploration_method: ExplorationStrategy = exploration_method
        super().__init__(data=data, model=model, opt=opt, **kwargs)
        self.ddpg_trainers = listify(trainers)
        for t in self.ddpg_trainers: self.callbacks.append(t(self))

    def predict(self, element, **kwargs):
        with torch.no_grad():
            training = self.model.training
            if element.shape[0] == 1: self.model.eval()
            pred = self.model(element)
            if training: self.model.train()
        return self.exploration_method.perturb(pred.detach().cpu().numpy(), self.data.action.action_space)

    def interpret_q(self, item):
        with torch.no_grad():
            return self.model.interpret_q(item).cpu().numpy().item()


class BaseDDPGTrainer(LearnerCallback):
    def __init__(self, learn):
        super().__init__(learn)
        self.max_episodes = 0
        self.episode = 0
        self.iteration = 0
        self.copy_over_frequency = 3

    @property
    def learn(self) -> DDPGLearner:
        return self._learn()

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs

    def on_epoch_begin(self, epoch, **kwargs: Any):
        self.episode = epoch
        self.iteration = 0

    def on_loss_begin(self, **kwargs: Any):
        """Performs tree updates, exploration updates, and model optimization."""
        if self.learn.model.training: self.learn.memory.update(item=self.learn.data.x.items[-1])
        self.learn.exploration_method.update(self.episode, max_episodes=self.max_episodes, explore=self.learn.model.training)
        if not self.learn.warming_up:
            samples: List[MDPStep] = self.memory.sample(self.learn.data.bs)
            post_optimize = self.learn.model.optimize(samples)
            if self.learn.model.training:
                self.learn.memory.refresh(post_optimize=post_optimize)
                self.learn.model.target_copy_over()
            self.iteration += 1


def create_ddpg_model(data: MDPDataBunch, base_arch: DDPGModule, layers=None, ignore_embed=False, channels=None,
                     opt=torch.optim.RMSprop, loss_func=None, **kwargs):
    bs, state, action = data.bs, data.state, data.action
    nc, w, h, n_conv_blocks = -1, -1, -1, [] if state.mode == FEED_TYPE_STATE else ifnone(channels, [32, 32, 32])
    if state.mode == FEED_TYPE_IMAGE: nc, w, h = state.s.shape[3], state.s.shape[2], state.s.shape[1]
    _layers = ifnone(layers, [400, 200] if len(n_conv_blocks) == 0 else [200, 200])
    if ignore_embed or np.any(state.n_possible_values == np.inf) or state.mode == FEED_TYPE_IMAGE: emb_szs = []
    else: emb_szs = [(d+1, int(fastai.tabular.data.emb_sz_rule(d))) for d in state.n_possible_values.reshape(-1, )]
    ao = int(action.taken_action.shape[1])
    model = base_arch(ni=state.s.shape[1], ao=ao, layers=_layers, emb_szs=emb_szs, n_conv_blocks=n_conv_blocks,
                      nc=nc, w=w, h=h, opt=opt, loss_func=loss_func, **kwargs)
    return model


ddpg_config = {
    DDPGModule: BaseDDPGTrainer
}


def ddpg_learner(data: MDPDataBunch, model, memory: ExperienceReplay, exploration_method: ExplorationStrategy,
                trainers=None, **kwargs):
    trainers = ifnone(trainers, ddpg_config[model.__class__])
    return DDPGLearner(data, model, memory, exploration_method, trainers, **kwargs)
