import os
import pickle
from copy import copy
from dataclasses import dataclass, field
from functools import partial
from itertools import cycle, product, islice
from math import floor
from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from fastai.basic_train import *
from fastai.sixel import plot_sixel
from fastai.train import Interpretation, torch, DatasetType, defaults, ifnone, warn
from fastprogress.fastprogress import IN_NOTEBOOK
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from fast_rl.core.data_block import MDPList, FEED_TYPE_IMAGE


def array_flatten(array):
    return [item for sublist in array for item in sublist]


def mplfig_to_npimage(fig):
    """
    Note, as of moviepy 1.0.1 there is an ugly depreciation warning when called the native binding in:
    `from moviepy.video.io.bindings import mplfig_to_npimage`

    This fixes the warning. Will hopefully be able to remove in a future datte.

    Converts a matplotlib figure to a RGB frame after updating the canvas"""
    #  only the Agg backend now supports the tostring_rgb function
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw() # update/draw the elements

    # get the width and the height to resize the matrix
    l,b,w,h = canvas.figure.bbox.bounds
    w, h = int(w), int(h)

    #  exports the canvas to a string buffer and then to a numpy nd.array
    buf = canvas.tostring_rgb()
    image= np.frombuffer(buf,dtype=np.uint8)
    return image.reshape(h,w,3)


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


class EpisodeNotAvailable(Exception): pass


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


@dataclass
class Gif:
    frames: np.array
    episode: int
    animation = None
    _frame_counter = 0
    _write_counter=0

    def __post_init__(self):
        if type(self.frames) is list: self.frames = np.concatenate(self.frames, axis=0)

    def _make_frame(self, t, frames, axes, fig, title, fps, matplot_to_np_fn):
        axes.clear()
        fig.suptitle(title + f' frame {floor(t * fps)}')
        axes.imshow(frames[floor(t * fps)] / 255)
        return matplot_to_np_fn(fig)

    def get_gif(self, default_fps=15, frame_skip=None):
        if frame_skip is not None: self.frames = self.frames[::frame_skip]
        try:
            from moviepy.video.VideoClip import VideoClip
            from moviepy.video.io.VideoFileClip import VideoFileClip
            from moviepy.video.io.html_tools import ipython_display

            fig, ax = plt.subplots()
            clip = VideoClip(partial(self._make_frame, frames=self.frames, axes=ax, fig=fig, fps=default_fps,
                                     matplot_to_np_fn=mplfig_to_npimage, title=f'Episode {self.episode}'),
                                     duration=(self.frames.shape[0] / default_fps)-1)
            plt.close(fig)
            return clip

        except ImportError:
            raise ImportError('Package: `moviepy` is not installed. You can install it via: `pip install moviepy`')

    def plot(self, fps=15, original_fps=15, cache_animation=True):
        if cache_animation or self.animation is None: self.animation = self.get_gif(original_fps)
        try:
            from moviepy.video.io.html_tools import ipython_display

            if not IN_NOTEBOOK: raise NotImplemented('Please use in a jupyter notebook or instead of `plot()` \n'
                                                     'call write("somefilename")')
            else: return ipython_display(self.animation, loop=True, autoplay=True, fps=fps)

        except ImportError:
            raise ImportError('Package: `moviepy` is not installed. You can install it via: `pip install moviepy`')


    def write(self, filename: str, include_episode=True, cache_animation=False, fps=15, original_fps=15,frame_skip=None):
        if self._write_counter>5:self._write_counter=0
        else:                    self._write_counter+=1
        try:
            if not cache_animation or self.animation is None: self.animation = self.get_gif(original_fps, frame_skip)
            if filename.__contains__('.gif'): filename = filename.replace('.gif', '')
            if include_episode: filename += f'_episode_{self.episode}'
            self.animation.write_gif(f"{filename}.gif", fps=fps)
        except RuntimeError as e:
            if self._write_counter>=5:
                warn(f'After 5 attempts, was unable to create gif: {str(e)}')
                return
            self.write(filename=filename,include_episode=include_episode,cache_animation=cache_animation,fps=fps,
                original_fps=original_fps,frame_skip=frame_skip)


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

    def frames(self, items):
        return [
            _.s.detach().cpu().numpy() if self.ds.feed_type == FEED_TYPE_IMAGE else _.alt_s_prime.detach().cpu().numpy()
            for _ in items]

    def generate_gif(self, episode: Union[None, list, int] = None) -> Union[Gif, List[Gif]]:
        full_episodes = list(set([k for k in self.ds.x.info if not self.ds.x.info[k][1]]) - {-1})
        if episode is None:
            episode = full_episodes
        elif episode == -1:
            episode = [list(sorted(full_episodes))[-1]]
        elif (type(episode) is not list and episode not in full_episodes) or \
                (type(episode) is list and any([e not in full_episodes for e in episode])):
            prefix = 'Some Episodes' if type(episode) is list else 'Episode'
            raise EpisodeNotAvailable(f'{prefix} {episode} not found in {full_episodes}. \nNote that due to the\n'
                                      f'memory manager, memory that is not related to the agent\'s training will be\n'
                                      f'deallocated. One of the first things deallocated are the rendered images\n'
                                      f'associated with each step since images typically take up large amounts of\n'
                                      f'memory. This then means there are fewer episodes that you can get full\n'
                                      f'gifs of. If this is not acceptable, make sure to change the memory management\n'
                                      f'strategy in the MDPDataBunch. ')
        elif type(episode) is not list and episode in full_episodes:
            episode = [episode]
        else:
            ValueError(f'Something happened that should not have happened, check your datatypes {episode}')

        gifs = [Gif(frames=self.frames(self.ds.x.filter_by_episode(e)), episode=e) for e in episode]
        return gifs[0] if len(gifs) == 1 else gifs


@dataclass
class GroupAgentInterpretation(object):
    groups: List[GroupField] = field(default_factory=list)
    in_notebook: bool = IN_NOTEBOOK

    @property
    def analysis(self):
        if not self.in_notebook:
            return [g.analysis for g in self.groups]
        else:
            return pd.DataFrame([{'name': g.unique_tuple, **g.analysis} for g in self.groups])

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
            actual += np.flip([np.cumsum(raw_actual[i:])[-1] for i in range(len(raw_actual))]).reshape(-1, ).tolist()
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
