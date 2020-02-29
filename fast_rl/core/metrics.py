from collections import deque

import torch
from fastai.basic_train import LearnerCallback, Any, ifnone
from fastai.callback import Callback, is_listy, add_metrics
import numpy as np


class EpsilonMetric(LearnerCallback):
    _order = -20  # Needs to run before the recorder

    def __init__(self, learn):
        super().__init__(learn)
        self.epsilon = 0
        if not hasattr(self.learn, 'exploration_method'):
            raise ValueError('Your model is not using an exploration strategy! Please use epsilon based exploration')
        if not hasattr(self.learn.exploration_method, 'epsilon'):
            raise ValueError('Please use epsilon based exploration (should have an epsilon field)')

    # noinspection PyUnresolvedReferences
    def on_train_begin(self, **kwargs):
        self.learn.recorder.add_metric_names(['epsilon'])

    def on_epoch_end(self, last_metrics, **kwargs):
        self.epsilon = self.learn.exploration_method.epsilon
        if last_metrics and last_metrics[-1] is None: del last_metrics[-1]
        return add_metrics(last_metrics, [float(self.epsilon)])

class RewardMetric(LearnerCallback):
    _order = -20

    def __init__(self, learn):
        super().__init__(learn)
        self.train_reward, self.valid_reward = [], []

    def on_epoch_begin(self, **kwargs:Any):
        self.train_reward, self.valid_reward = [], []

    def on_batch_end(self, **kwargs: Any):
        if self.learn.model.training: self.train_reward.append(self.learn.data.train_ds.item.reward.cpu().numpy()[0][0])
        elif not self.learn.recorder.no_val: self.valid_reward.append(self.learn.data.valid_ds.item.reward.cpu().numpy()[0][0])

    def on_train_begin(self, **kwargs):
        metric_names = ['train_reward'] if self.learn.recorder.no_val else ['train_reward', 'valid_reward']
        self.learn.recorder.add_metric_names(metric_names)

    def on_epoch_end(self, last_metrics, **kwargs: Any):
        return add_metrics(last_metrics, [sum(self.train_reward), sum(self.valid_reward)])

class RollingRewardMetric(LearnerCallback):
    _order = -20

    def __init__(self, learn,rolling_size=None):
        super().__init__(learn)
        self.rolling_sz=ifnone(rolling_size,self.learn.data.bs)
        self.train_reward, self.valid_reward = [], []
        self.train_rolling_reward,self.valid_rolling_reward=deque([],maxlen=self.rolling_sz), deque([],maxlen=self.rolling_sz)

    def on_epoch_begin(self, **kwargs:Any):
        self.train_reward, self.valid_reward = [], []

    def on_batch_end(self, **kwargs: Any):
        if self.learn.model.training: self.train_reward.append(self.learn.data.train_ds.item.reward.cpu().numpy()[0][0])
        elif not self.learn.recorder.no_val: self.valid_reward.append(self.learn.data.valid_ds.item.reward.cpu().numpy()[0][0])

    def on_train_begin(self, **kwargs):
        metric_names = ['train_rolling_reward'] if self.learn.recorder.no_val else ['train_rolling_reward', 'valid_rolling_reward']
        self.learn.recorder.add_metric_names(metric_names)

    def on_epoch_end(self, last_metrics, **kwargs: Any):
        self.train_rolling_reward.append(sum(self.train_reward))
        self.valid_rolling_reward.append(sum(self.valid_reward))
        return add_metrics(last_metrics, [np.average(self.train_rolling_reward), np.average(self.valid_rolling_reward)])

