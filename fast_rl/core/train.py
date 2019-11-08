import pickle
from copy import copy
from pathlib import Path

from torch.distributions import Normal
from dataclasses import dataclass, field
from fastai.basic_train import *
from fastai.sixel import plot_sixel
from fastai.train import Interpretation, torch, DatasetType, defaults, ifnone
import matplotlib.pyplot as plt
from fastprogress.fastprogress import IN_NOTEBOOK
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Union, List
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

    @property
    def analysis(self):
        return [g.analysis for g in self.groups]

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
                           smooth_groups: Union[None, float] = None, figsize=(5, 5)):
        groups = self.filter_by(per_episode, 'reward')
        if smooth_groups is not None: [g.smooth(smooth_groups) for g in groups]
        unique_values = list(set([g.unique_tuple for g in groups]))
        colors = list(islice(cycle(plt.rcParams['axes.prop_cycle'].by_key()['color']), len(unique_values)))
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # type: Figure, Axes

        for grouped, c in zip(self.group_by(groups, unique_values), colors):
            min_len = min([len(v.values) for v in grouped])
            min_b = np.min([v.values[:min_len] for v in grouped], axis=0)
            max_b = np.max([v.values[:min_len] for v in grouped], axis=0)
            # TODO fit function sometimes does +1 more episodes...  WHY?
            overflow = [v.values for v in grouped if len(v.values) - min_len > 2]

            ax.plot(min_b, c=c, label=f'{grouped[0].meta} {grouped[0].model}')
            ax.plot(max_b, c=c)
            for v in overflow: ax.plot(v, c=c)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax.fill_between(list(range(min_len)), min_b, max_b, where=max_b > min_b, color=c, alpha=0.3)

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
