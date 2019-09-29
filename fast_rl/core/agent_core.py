import copy
import math
import random
from collections import deque
from functools import partial
from math import ceil
from typing import List, Optional

import gym
import numpy as np
import torch
from fastai.basic_train import *
from fastai.basic_train import BasicLearner, CallbackList, OptMetrics, master_bar, is_listy, first_el, to_np
from fastai.callback import CallbackHandler
from fastprogress import progress_bar

from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice
from fast_rl.core.data_structures import SumTree


class ExplorationStrategy:
    def __init__(self, do_exploration: bool):
        self.do_exploration = do_exploration

    def perturb(self, action, action_space):
        """
        Base method just returns the action. Subclass, and change to return randomly / augmented actions.

        Should use `do_exploration` field. It is recommended that when you subclass / overload, you allow this field
        to completely bypass these actions.

        Args:
            action:
            action_space (gym.Space): The original gym space. Should contain information on the action type, and
            possible convenience methods for random action selection.

        Returns:

        """
        _ = action_space
        return action

    def update(self, do_exploration, **kwargs):
        self.do_exploration = do_exploration


class GreedyEpsilon(ExplorationStrategy):
    def __init__(self, epsilon_start, epsilon_end, decay, start_episode=0, end_episode=0, **kwargs):
        super().__init__(**kwargs)
        self.end_episode = end_episode
        self.start_episode = start_episode
        self.decay = decay
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon = self.epsilon_start
        self.steps_done = 0

    def perturb(self, action, action_space: gym.Space):
        """
        TODO for now does random discrete selection. Move to continuous soon.

        Args:
            action:
            action_space:

        Returns:
        """
        if np.random.random() < self.epsilon:
            return action_space.sample()
        else:
            return action

    def update(self, current_episode, end_episode=0, **kwargs):
        super(GreedyEpsilon, self).update(**kwargs)
        if self.do_exploration:
            self.end_episode = end_episode
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           math.exp(-1. * (self.steps_done * self.decay))
            self.steps_done += 1


class OrnsteinUhlenbeck(GreedyEpsilon):
    def __init__(self, epsilon_start, epsilon_end, decay, **kwargs):
        """

        References:
            [1] From https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
            [2] Cumulatively based on

        Args:
            epsilon_start:
            epsilon_end:
            decay:
            **kwargs:
        """
        super().__init__(epsilon_start, epsilon_end, decay, **kwargs)


class Experience:
    def __init__(self, memory_size):
        self.max_size = memory_size
        self.callbacks = []

    def sample(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass

    def refresh(self, **kwargs):
        pass


class ExperienceReplay(Experience):
    def __init__(self, memory_size):
        """
        Basic store-er of state space transitions for training agents.

        References:
            [1] Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning."
            arXiv preprint arXiv:1312.5602 (2013).

        Args:
            memory_size (int): Max N samples to store
        """
        super().__init__(memory_size)
        self.max_size = memory_size
        self.memory = deque(maxlen=memory_size)  # type: List[MarkovDecisionProcessSlice]

    def __len__(self):
        return len(self.memory)

    def sample(self, batch, **kwargs):
        if len(self.memory) < batch: return self.memory
        return random.sample(self.memory, batch)

    def update(self, item, **kwargs):
        self.memory.append(copy.deepcopy(item))


class PriorityExperienceReplayCallback(LearnerCallback):
    def on_train_begin(self, **kwargs):
        self.learn.model.loss_func = partial(self.learn.model.memory.handle_loss,
                                             base_function=self.learn.model.loss_func)


class PriorityExperienceReplay(Experience):

    def handle_loss(self, y, y_hat, base_function):
        return (base_function(y, y_hat) * torch.from_numpy(self.priority_weights).float()).mean().float()

    def __init__(self, memory_size, batch_size=64, epsilon=0.001, alpha=0.6, beta=0.5):
        """
        Prioritizes sampling based on samples requiring the most learning.

        References:
            [1] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

        Args:
            batch_size (int): Size of sample, and thus size of expected index update.
            alpha (float): Changes the sampling behavior 1 (non-uniform) -> 0 (uniform)
            epsilon (float): Keeps the probabilities of items from being 0
            memory_size (int): Max N samples to store
        """
        super().__init__(memory_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.b_inc = -0.00001
        self.priority_weights = np.zeros(self.batch_size, dtype=float)
        self.epsilon = epsilon
        self.memory = SumTree(self.max_size)
        self.callbacks = [PriorityExperienceReplayCallback]
        # When sampled, store the sample indices for refresh.
        self._indices = np.zeros(self.batch_size, dtype=int)

    def __len__(self):
        return self.memory.n_entries

    def refresh(self, post_optimize, **kwargs):
        if post_optimize is not None:
            self.memory.update(self._indices.astype(int), np.abs(post_optimize['td_error']) + self.epsilon)

    def sample(self, batch, **kwargs):
        self.beta = np.min([1., self.beta + self.b_inc])
        # ranges = np.linspace(0, self.memory.total(), num=ceil(self.memory.total() / self.batch_size))
        ranges = np.linspace(0, ceil(self.memory.total() / self.batch_size), num=self.batch_size + 1)
        uniform_ranges = [np.random.uniform(ranges[i], ranges[i + 1]) for i in range(len(ranges) - 1)]
        self._indices, weights, samples = self.memory.batch_get(uniform_ranges)
        self.priority_weights = self.memory.anneal_weights(weights, self.beta)
        return samples

    def update(self, item, **kwargs):
        """
        Updates the memory of PER.

        Assigns maximal priority per [1] Alg:1, thus guaranteeing that sample being visited once.

        Args:
            item:

        Returns:

        """
        maximal_priority = self.alpha
        self.memory.add(np.abs(maximal_priority) + self.epsilon, item)


class HindsightExperienceReplay(Experience):
    def __init__(self, memory_size):
        """

        References:
            [1] Andrychowicz, Marcin, et al. "Hindsight experience replay."
            Advances in Neural Information Processing Systems. 2017.

        Args:
            memory_size:
        """
        super().__init__(memory_size)


def loss_batch(model, cb_handler: Optional[CallbackHandler]):
    """ TODO will be different. Is there anything extra needed from here? """
    if model.out is not None: cb_handler.on_loss_begin(model.out)
    if model.loss is None: return None
    cb_handler.on_backward_begin(model.loss)
    cb_handler.on_backward_end()
    cb_handler.on_step_end()
    return model.loss.detach().cpu()


def validate(learn, dl, cb_handler: Optional[CallbackHandler] = None,
             pbar=None, average=True, n_batch: Optional[int] = None):
    learn.model.eval()
    with torch.no_grad():
        val_losses, nums = [], []
        if cb_handler: cb_handler.set_dl(dl)
        # TODO 1st change: in fit function, original uses xb, yb. Maybe figure out what 2nd value to include?
        for element in progress_bar(dl, parent=pbar):
            learn.data.valid_ds.actions = learn.predict(element)
            if cb_handler: element = cb_handler.on_batch_begin(element, learn.data.valid_ds.actions, train=False)
            val_loss = loss_batch(learn.model, cb_handler=cb_handler)
            if val_loss is None: continue
            val_losses.append(val_loss)
            r = dl.actions if is_listy(dl.actions) else np.array([dl.actions])
            nums.append(first_el(r).shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums) >= n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average and val_losses: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else: return val_losses


def fit(epochs: int, learn: BasicLearner, callbacks: Optional[CallbackList] = None, metrics: OptMetrics = None):
    """
    Takes a RL Learner and trains it on a given environment.

    Important behavior notes:

    - The original Fastai fit function outputs the loss for every epoch. Since many RL models need to fill a memory buffer before optimization, the fit function for epoch 0 will run multiple episodes until the memory is filled.
    - **So when you see the agent running through episodes without output, please note that it is most likely filling the memory first before properly printing.**

    Args:
        epochs:
        learn:
        callbacks:
        metrics:

    Returns:

    """
    assert len(learn.data.train_dl) != 0, f"""Your training dataloader is empty, can't train a model. Use a smaller 
        batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""
    # Since CallbackHandler is a dataclass, these input fields will be automatically populated via
    # a default __init__
    cb_handler = CallbackHandler(callbacks, metrics)
    cb_handler.state_dict['skip_validate'] = learn.data.empty_val
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)
    # Note that metrics in the on_train begin method need a name field.
    exception = False
    loss = None
    try:
        for epoch in pbar:
            learn.model.train()
            cb_handler.set_dl(learn.data.train_dl)
            cb_handler.on_epoch_begin()
            # TODO 1st change: While the loss is None, the model's memory has not filled  and / or is not ready.
            while loss is None:
                # TODO 2nd change: in fit function, original uses xb, yb. Maybe figure out what 2nd value to include?
                for element in progress_bar(learn.data.train_dl, parent=pbar):
                    # TODO 3rd change: get the action for the given state. Move to on batch begin callback?
                    learn.data.train_ds.actions = learn.predict(element)
                    cb_handler.on_batch_begin(element, learn.data.train_ds.actions)
                    # TODO 4th change: loss_batch is way simpler... What is a batch to be defined as?
                    loss = loss_batch(learn.model, cb_handler)
                    if cb_handler.on_batch_end(loss): break

            loss = None
            if not cb_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(learn, learn.data.valid_dl, cb_handler=cb_handler, pbar=pbar)
            else: val_loss = None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)
