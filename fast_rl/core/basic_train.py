from fastai.basic_train import Learner


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
        self.callback_fns += self.model.learner_callbacks + self.data.train_ds.callback

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
