from copy import copy

from fastai.basic_train import Learner, DatasetType, ifnone, defaults
from fastai.sixel import plot_sixel
from fastai.train import Interpretation
import matplotlib.pyplot as plt
from fastprogress.fastprogress import IN_NOTEBOOK
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Tuple, Union, List
import numpy as np
from matplotlib.ticker import MaxNLocator

from fast_rl.core.MarkovDecisionProcess import MDPList


def array_flatten(array):
    return [item for sublist in array for item in sublist]


class AgentInterpretation(Interpretation):
    def __init__(self, learn: Learner, ds_type: DatasetType = DatasetType.Valid):
        super().__init__(learn, None, None, None, ds_type=ds_type)

    def group_by_episode(self, items: MDPList, episodes: list):
        ils = [copy(items).filter_by_func(lambda x: x.episode in episodes and x.episode == ep) for ep in episodes]
        return [il for il in ils if len(il.items) != 0]

    def process_values(self, values: Union[list, List[list]], squash_episodes, cumulative):
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

    def get_values(self, il: MDPList, value_name, per_episode=False):
        if per_episode:
            return [self.get_values(i, value_name) for i in self.group_by_episode(il, list(il.info.keys()))]
        return [i.obj[value_name] for i in il.items]

    def line_plot(self, values: Union[list, List[list]], figsize=(5, 5), cumulative=False, squash_episodes=True):
        fig, ax = plt.subplots(1, 1, figsize=figsize)  # type: Figure, Axes
        ax.plot(values)

        ax.set_title(f'Rewards over {"episodes" if squash_episodes else "iterations"}')
        ax.set_ylabel(f'{"Cumulative " if cumulative else ""}Rewards')
        ax.set_xlabel("Iterations " if not squash_episodes else "Episodes")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return fig

    def plot_rewards(self, per_episode=False, return_fig: bool = None, **kwargs):
        values = self.get_values(self.ds.x, 'reward', per_episode)
        processed_values = self.process_values(values, **kwargs)
        fig = self.line_plot(processed_values, **kwargs)

        if ifnone(return_fig, defaults.return_fig): return fig
        if not IN_NOTEBOOK: plot_sixel(fig)
