import pickle
from copy import copy
from functools import partial
from pathlib import Path

import scipy.stats as st
from torch.distributions import Normal
from dataclasses import dataclass, field
from fastai.basic_train import *
from fastai.sixel import plot_sixel
from fastai.train import Interpretation, torch, DatasetType, defaults, ifnone, warn
import matplotlib.pyplot as plt
from fastprogress.fastprogress import IN_NOTEBOOK
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Union, List
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
from itertools import cycle, product, permutations, combinations, combinations_with_replacement, islice

from fast_rl.core.data_block import MDPList


def array_flatten(array):
    return [item for sublist in array for item in sublist]


def cumulate_squash(values: Union[list, List[list]], squash_episodes=False, cumulative=False):
    if isinstance(values[0], list):
        if squash_episodes:
            if cumulative:
                values = [np.max(np.cumsum(episode)) for episode in values]
            else:
                values = [np.max(episode) for episode in values]
        else:
            if cumulative:
                values = [np.cumsum(episode) for episode in array_flatten(values)]
            else:
                values = array_flatten(values)
    else:
        if cumulative: values = np.cumsum(values)
    return values


def group_by_episode(items: MDPList, episodes: list):
    ils = [copy(items).filter_by_func(lambda x: x.episode in episodes and x.episode == ep) for ep in episodes]
    return [il for il in ils if len(il.items) != 0]


def smooth(v, smoothing_const): return np.convolve(v, np.ones(smoothing_const), 'same') / smoothing_const


@dataclass
class GroupField:
    values: list
    model: str
    meta: str
    value_type: str
    per_episode: bool

    @property
    def analysis(self):
        return {
            'average': np.average(self.values),
            'max': np.max(self.values),
            'min': np.min(self.values),
            'type': self.value_type
        }

    @property
    def unique_tuple(self): return self.model, self.meta, self.value_type

    def __eq__(self, other):
        comp_tuple = other.unique_tuple if isinstance(other, GroupField) else other
        return all([self.unique_tuple[i] == comp_tuple[i] for i in range(len(self.unique_tuple))])

    def smooth(self, smooth_groups): self.values = smooth(self.values, smooth_groups)


class AgentInterpretation(Interpretation):
    def __init__(self, learn: Learner, ds_type: DatasetType = DatasetType.Valid, close_env=True):
        super().__init__(learn, None, None, None, ds_type=ds_type)
        self.groups = []
        if close_env: self.ds.env.close()

    def get_values(self, il: MDPList, value_name, per_episode=False):
        if per_episode:
            return [self.get_values(i, value_name) for i in group_by_episode(il, list(il.info.keys()))]
        return [i.obj[value_name] for i in il.items]

    def line_figure(self, values: Union[list, List[list]], figsize=(5, 5), cumulative=False, per_episode=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # type: Figure, Axes
        ax.plot(values)

        ax.set_title(f'Rewards over {"episodes" if per_episode else "iterations"}')
        ax.set_ylabel(f'{"Cumulative " if cumulative else ""}Rewards')
        ax.set_xlabel("Episodes " if per_episode else "Iterations")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return fig

    def plot_rewards(self, per_episode=False, return_fig: bool = None, group_name=None, cumulative=False, no_show=False,
                     smooth_const: Union[None, float] = None, **kwargs):
        values = self.get_values(self.ds.x, 'reward', per_episode)
        processed_values = cumulate_squash(values, squash_episodes=per_episode, cumulative=cumulative, **kwargs)
        if group_name: self.groups.append(GroupField(processed_values, self.learn.model.name, group_name, 'reward',
                                                     per_episode))
        if no_show: return
        if smooth_const: processed_values = smooth(processed_values, smooth_const)
        fig = self.line_figure(processed_values, per_episode=per_episode, cumulative=cumulative, **kwargs)

        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)

    def to_group_agent_interpretation(self):
        interp = GroupAgentInterpretation()
        interp.add_interpretation(self)
        return interp


@dataclass
class GroupAgentInterpretation(object):
    groups: List[GroupField] = field(default_factory=list)
    in_notebook: bool = IN_NOTEBOOK

    @property
    def analysis(self):
        if not self.in_notebook: return [g.analysis for g in self.groups]
        else: return pd.DataFrame([{'name': g.unique_tuple, **g.analysis} for g in self.groups])

    def append_meta(self, post_fix):
        r""" Useful before calling `to_pickle` if you want this set to be seen differently from future runs."""
        for g in self.groups: g.meta = g.meta + post_fix
        return self

    def filter_by(self, per_episode, value_type):
        return copy([g for g in self.groups if g.value_type == value_type and g.per_episode == per_episode])

    def group_by(self, groups, unique_values):
        for comp_tuple in unique_values: yield [g for g in groups if g == comp_tuple]

    def add_interpretation(self, interp):
        self.groups += interp.groups

    def plot_reward_bounds(self, title=None, return_fig: bool = None, per_episode=False,
                           smooth_groups: Union[None, float] = None, figsize=(5, 5), show_average=False,
                           hide_edges=False):
        groups = self.filter_by(per_episode, 'reward')
        if smooth_groups is not None: [g.smooth(smooth_groups) for g in groups]
        unique_values = list(set([g.unique_tuple for g in groups]))
        colors = list(islice(cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']), len(unique_values)))
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # type: Figure, Axes

        for grouped, c in zip(self.group_by(groups, unique_values), colors):
            min_len = min([len(v.values) for v in grouped])
            min_b = np.min([v.values[:min_len] for v in grouped], axis=0)
            max_b = np.max([v.values[:min_len] for v in grouped], axis=0)
            if show_average:
                average = np.average([v.values[:min_len] for v in grouped], axis=0)
                ax.plot(average, c=c, linestyle=':')
            # TODO fit function sometimes does +1 more episodes...  WHY?
            overflow = [v.values for v in grouped if len(v.values) - min_len > 2]

            if not hide_edges:
                ax.plot(min_b, c=c)
                ax.plot(max_b, c=c)
                for v in overflow: ax.plot(v, c=c)

            ax.fill_between(list(range(min_len)), min_b, max_b, where=max_b > min_b, color=c, alpha=0.3,
                            label=f'{grouped[0].meta} {grouped[0].model}')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(ifnone(title, f'{"Per Episode" if per_episode else "Per Step"} Rewards'))
        ax.set_ylabel('Rewards')
        ax.set_xlabel(f'{"Episodes" if per_episode else "Steps"}')

        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)

    def to_pickle(self, root_path, name):
        if not os.path.exists(root_path): os.makedirs(root_path)
        pickle.dump(self, open(Path(root_path) / (name + ".pickle"), "wb"), pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls, root_path, name) -> 'GroupAgentInterpretation':
        return pickle.load(open(Path(root_path) / f'{name}.pickle', 'rb'))

    def merge(self, other):
        return __class__(self.groups + other.groups)


class GymMazeInterpretation(AgentInterpretation):
    def __init__(self, learn: Learner, **kwargs):
        super().__init__(learn, **kwargs)
        try:
            from gym_maze.envs import MazeEnv
            if not issubclass(self.learn.data.env.unwrapped.__class__, MazeEnv):
                raise NotImplemented('This learner was trained on an environment that is not a gym maze env.')
        except ImportError as e:
            warn(f'Could not import gym maze. Do you have it installed? Full error: {e}')
        self.bounds = self.learn.data.state.bounds

    def eval_action(self, action_raw, index=-1):
        action_raw = action_raw[0]  # Remove batch dim
        return torch.max(action_raw).numpy().item() if index == -1 else action_raw[index].numpy().item()

    def heat_map(self, action):
        action_eval_fn = partial(self.eval_action, index=action)
        state_bounds = list(product(*(np.arange(r[0], r[1]) for r in zip(self.bounds.min, self.bounds.max))))
        # if min is -1, then it is an extra dimension, so multiply by -1 so that the dim in max + 1.
        heat_map = np.zeros(shape=tuple(self.bounds.max + -1 * self.bounds.min))
        action_map = np.zeros(shape=tuple(self.bounds.max + -1 * self.bounds.min))
        for state in state_bounds:
            with torch.no_grad():
                heat_map[state] = action_eval_fn(self.learn.model(torch.Tensor(data=state).unsqueeze(0).long()))
                action_map[state] = self.learn.predict(torch.Tensor(data=state).unsqueeze(0).long())
        return heat_map, action_map

    def add_text_to_image(self, ax, action_map):
        x_start, y_start, x_end, y_end, size = 0, 0, action_map.shape[0], action_map.shape[1], 1
        # Add the text
        jump_x = size
        jump_y = size
        x_positions = np.linspace(start=x_start, stop=x_end, num=x_end, endpoint=False) - 1
        y_positions = np.linspace(start=y_start, stop=y_end, num=y_end, endpoint=False) - 1

        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = action_map[y_index, x_index]
                text_x = x + jump_x
                text_y = y + jump_y
                ax.text(text_x, text_y, int(label), color='black', ha='center', va='center')

    def plot_heat_map(self, action=-1, figsize=(7, 7), return_fig=None):
        exploring = self.learn.exploration_method.explore
        self.learn.exploration_method.explore = False
        heat_map, chosen_actions = self.heat_map(action)
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # type: Figure, Axes
        im = ax.imshow(heat_map)
        fig.colorbar(im)
        title = f'Heat mapped values {"maximum" if action == -1 else "for action " + str(action)}'
        title += '\nText: Chosen action for a given state'
        ax.set_title(title)
        self.add_text_to_image(ax, chosen_actions)
        self.learn.exploration_method.explore = exploring

        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)


class QValueInterpretation(AgentInterpretation):
    def __init__(self, learn: Learner, **kwargs):
        super().__init__(learn, **kwargs)
        self.items = self.learn.data.x if len(self.learn.data.x) != 0 else self.learn.memory.memory

    def normalize(self, item: np.array):
        if np.max(item) - np.min(item) != 0:
            return np.divide(item + np.min(item), np.max(item) - np.min(item))
        else:
            item.fill(1)
            return item

    def q(self, items):
        actual, predicted = [], []
        episode_partition = [[i for i in items.items if i.episode == key] for key in items.info]

        for ei in episode_partition:
            if not ei: continue
            raw_actual = [ei[i].reward.cpu().numpy().item() for i in np.flip(np.arange(len(ei)))]
            actual += np.flip([np.cumsum(raw_actual[i:])[-1] for i in range(len(raw_actual))]).reshape(-1,).tolist()
            for item in ei: predicted.append(self.learn.interpret_q(item))

        return self.normalize(actual), self.normalize(predicted)

    def plot_q(self, figsize=(8, 8), return_fig=None):
        r"""
        Heat maps the density of actual vs estimated q v. Good reference for this is at [1].

        References:
            [1] "Simple Example Of 2D Density Plots In Python." Medium. N. p., 2019. Web. 31 Aug. 2019.
            https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67

        Returns:

        """
        q_action, q_predicted = self.q(self.items)
        # Define the borders
        deltaX = (np.max(q_action) - np.min(q_action)) / 10
        deltaY = (np.max(q_predicted) - np.min(q_predicted)) / 10
        xmin = np.min(q_action) - deltaX
        xmax = np.max(q_action) + deltaX
        ymin = np.min(q_predicted) - deltaY
        ymax = np.max(q_predicted) + deltaY
        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([q_action, q_predicted])

        kernel = st.gaussian_kde(values)

        f = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('Actual Returns')
        ax.set_ylabel('Estimated Q')
        ax.set_title('2D Gaussian Kernel Q Density Estimation')

        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)