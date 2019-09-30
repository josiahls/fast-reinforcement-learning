import torch
from fastai.basic_train import LearnerCallback
from fastai.callback import Callback, is_listy, add_metrics


class EpsilonMetric(LearnerCallback):
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn):
        super().__init__(learn)
        self.epsilon = 0
        if not hasattr(self.learn.model, 'exploration_strategy'):
            raise ValueError('Your model is not using an exploration strategy! Please use epsilon based exploration')
        if not hasattr(self.learn.model.exploration_strategy, 'epsilon'):
            raise ValueError('Please use epsilon based exploration (should have an epsilon field)')

    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['epsilon'])

    def on_epoch_end(self, last_metrics, **kwargs):
        self.epsilon = self.learn.model.exploration_strategy.epsilon
        if last_metrics and last_metrics[-1] is None: del last_metrics[-1]
        return add_metrics(last_metrics, [float(self.epsilon)])
