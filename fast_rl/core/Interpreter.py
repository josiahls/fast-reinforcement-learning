import io
from functools import partial
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
from PIL import Image
from fastai.train import Interpretation, DatasetType, copy
from gym.spaces import Box
from itertools import product
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from torch import nn

from fast_rl.core import Learner
from fast_rl.core.data_block import MarkovDecisionProcessSliceAlpha, FEED_TYPE_IMAGE


class AgentInterpretationAlpha(Interpretation):
    def __init__(self, learn: Learner, ds_type: DatasetType = DatasetType.Valid, base_chart_size=(20, 10)):
        """
        Handles converting a learner, and it's runs into useful human interpretable information.

        Notes:
            This class is called AgentInterpretationAlpha because it will overall get deprecated.
            The final working version will be called AgentInterpretation.

        Args:
            learn:
        """
        super().__init__(learn, None, None, None, ds_type=ds_type)
        self.current_animation = None
        plt.rcParams["figure.figsize"] = base_chart_size
        
    def _get_items(self, ignore=True):
        episodes = list(self.ds.x.info.keys())
        if ignore or len(episodes) == 0: return self.ds.x.items
        return [item for item in self.ds.x.items if item.episode in episodes]

    @classmethod
    def from_learner(cls, learn: Learner, ds_type: DatasetType = DatasetType.Valid, activ: nn.Module = None):
        raise NotImplementedError

    def normalize(self, item: np.array):
        if np.max(item) - np.min(item) != 0:
            return np.divide(item + np.min(item), np.max(item) - np.min(item))
        else:
            item.fill(1)
            return item

    def top_losses(self, k: int = None, largest=True):
        raise NotImplementedError

    def reward_heatmap(self, episode_slices: List[MarkovDecisionProcessSliceAlpha], action=None):
        """
        Takes a state_space and uses the agent to heat map rewards over the space.

        We first need to determine if the s space is discrete or discrete.

        Args:
            state_space:

        Returns:

        """
        if action is not None: action = torch.tensor(action).long()
        current_state_slice = [p for p in product(
            np.arange(min(self.ds.env.observation_space.low), max(self.ds.env.observation_space.high) + 1),
            repeat=len(self.ds.env.observation_space.high))]
        heat_map = np.zeros(np.add(self.ds.env.observation_space.high, 1))
        with torch.no_grad():
            for state in current_state_slice:
                if action is not None:
                    heat_map[state] = self.learn.model(torch.from_numpy(np.array(state)).unsqueeze(0))[0].gather(0, action)
                else:
                    self.learn.model.eval()
                    if self.learn.model.name == 'DDPG':
                        heat_map[state] = self.learn.model.critic_model(torch.cat((torch.from_numpy(np.array(state)).unsqueeze(0).float(), self.learn.model.action_model(torch.from_numpy(np.array(state)).unsqueeze(0).float())), 1))
                    else:
                        heat_map[state] = self.learn.model(torch.from_numpy(np.array(state)).unsqueeze(0))[0].max().numpy()
        return heat_map

    def plot_heatmapped_episode(self, episode, fig_size=(13, 5), action_index=None, return_heat_maps=False):
        """
        Generates plots of heatmapped s spaces for analyzing reward distribution.

        Currently only makes sense for grid based envs. Will be expecting gym_maze environments that are discrete.

        Returns:

        """
        if not str(self.ds.env.spec).__contains__('maze'):
            raise NotImplementedError('Currently only supports gym_maze envs that have discrete s spaces')
        if not isinstance(self.ds.state_size, Box):
            raise NotImplementedError('Currently only supports Box based s spaces with 2 dimensions')

        items = self._get_items()
        heat_maps = []

        # For each episode
        buffer = []
        episode = episode if episode != -1 else list(set([i.episode for i in items]))[-1]
        for item in [i for i in items if i.episode == episode]:
            buffer.append(item)
        heat_map = self.reward_heatmap(buffer, action=action_index)
        heat_maps.append((copy(heat_map), copy(buffer[-1]), copy(episode)))

        plots = []
        for single_heatmap in [heat_maps[-1]]:
            fig, ax = plt.subplots(1, 2, figsize=fig_size)
            fig.suptitle(f'Episode {episode}')
            ax[0].imshow(single_heatmap[1].to_one().data)
            im = ax[1].imshow(single_heatmap[0])
            ax[0].grid(False)
            ax[1].grid(False)
            ax[0].set_title('Final State Snapshot')
            ax[1].set_title('State Space Heatmap')
            fig.colorbar(im, ax=ax[1])

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(fig)
            buf.seek(0)
            # Create Image object
            plots.append(np.array(Image.open(buf))[:, :, :3])

        for plot in plots:
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.imshow(plot)
            plt.show()

        if return_heat_maps: return heat_maps

    def plot_episode(self, episode):
        items = self._get_items(False)  # type: List[MarkovDecisionProcessSliceAlpha]

        episode_counter = 0
        # For each episode
        buffer = []
        for item in items:
            buffer.append(item)
            if item.done:
                if episode_counter == episode:
                    break
                episode_counter += 1
                buffer = []

        plots = []
        with torch.no_grad():
            agent_reward_plots = [self.learn.model(torch.from_numpy(np.array(i.current_state))).max().numpy() for i in
                                  buffer]
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            fig.suptitle(f'Episode {episode}')
            ax.plot(agent_reward_plots)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Max Expected Reward from Agent')

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            # Closing the figure prevents it from being displayed directly inside
            # the notebook.
            plt.close(fig)
            buf.seek(0)
            # Create Image object
            plots.append(np.array(Image.open(buf))[:, :, :3])

        for plot in plots:
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.imshow(plot)
            plt.show()

    def get_agent_accuracy_density(self, items, episode_num=None):
        x = None
        y = None

        for episode in [_ for _ in list(set(mdp.episode for mdp in items)) if episode_num is None or episode_num == _]:
            subset = [item for item in items if item.episode == episode]
            state = np.array([_.current_state for _ in subset])
            result_state = np.array([_.result_state for _ in subset])

            prim_q_pred = self.learn.model(torch.from_numpy(state))
            target_q_pred = self.learn.model.target_net(torch.from_numpy(state).float())
            state_difference = (prim_q_pred - target_q_pred).sum(1)
            prim_q_pred = self.learn.model(torch.from_numpy(result_state))
            target_q_pred = self.learn.model.target_net(torch.from_numpy(result_state).float())
            result_state_difference = (prim_q_pred - target_q_pred).sum(1)

            x = state_difference if x is None else np.hstack((x, state_difference))
            y = result_state_difference if y is None else np.hstack((y, result_state_difference))

        return x, y

    def plot_agent_accuracy_density(self, episode_num=None):
        """
        Heat maps the density of actual vs estimated q v. Good reference for this is at [1].

        References:
            [1] "Simple Example Of 2D Density Plots In Python." Medium. N. p., 2019. Web. 31 Aug. 2019.
            https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67

        Returns:

        """
        items = self._get_items(False)  # type: List[MarkovDecisionProcessSliceAlpha]
        x, y = self.get_agent_accuracy_density(items, episode_num)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        fig.suptitle(f'{self.learn.model.name} for {self.ds.env.spec._env_name}')
        ax.set_ylabel('State / State Prime Q Value Deviation')
        ax.set_xlabel('Iterations')
        ax.plot(np.hstack([x, y]))
        plt.show()

    def get_q_density(self, items, episode_num=None):
        x = None
        y = None

        for episode in [_ for _ in list(set(mdp.episode for mdp in items)) if episode_num is None or episode_num == _]:
            subset = [item for item in items if item.episode == episode]
            r = np.array([_.reward for _ in subset])
            # Gets the total accumulated r over a single markov chain
            actual_returns = np.flip([np.cumsum(r)[i:][0] for i in np.flip(np.arange(len(r)))]).reshape(1, -1)
            estimated_returns = self.learn.model.interpret_q(subset).view(1, -1).numpy()
            x = actual_returns if x is None else np.hstack((x, actual_returns))
            y = estimated_returns if y is None else np.hstack((y, estimated_returns))

        return self.normalize(x), self.normalize(y)

    def plot_q_density(self, episode_num=None):
        """
        Heat maps the density of actual vs estimated q v. Good reference for this is at [1].

        References:
            [1] "Simple Example Of 2D Density Plots In Python." Medium. N. p., 2019. Web. 31 Aug. 2019.
            https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67

        Returns:

        """
        items = self._get_items(False)  # type: List[MarkovDecisionProcessSliceAlpha]
        x, y = self.get_q_density(items, episode_num)

        # Define the borders
        deltaX = (np.max(x) - np.min(x)) / 10
        deltaY = (np.max(y) - np.min(y)) / 10
        xmin = np.min(x) - deltaX
        xmax = np.max(x) + deltaX
        ymin = np.min(y) - deltaY
        ymax = np.max(y) + deltaY
        print(xmin, xmax, ymin, ymax)
        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x, y])

        kernel = st.gaussian_kde(values)

        f = np.reshape(kernel(positions).T, xx.shape)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('Actual Returns')
        ax.set_ylabel('Estimated Q')
        if episode_num is None:
            plt.title('2D Gaussian Kernel Q Density Estimation')
        else:
            plt.title(f'2D Gaussian Kernel Q Density Estimation for episode {episode_num}')
        plt.show()

    def plot_rewards_over_iterations(self, cumulative=False, return_rewards=False):
        items = self._get_items()
        r_iter = [el.reward[0] if np.ndim(el.reward) == 0 else np.average(el.reward) for el in items]
        if cumulative: r_iter = np.cumsum(r_iter)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        fig.suptitle(f'{self.learn.model.name} for {self.ds.env.spec._env_name}')
        ax.set_ylabel('Rewards' if not cumulative else 'Cumulative Rewards')
        ax.set_xlabel('Iterations')
        ax.plot(r_iter)
        plt.show()
        if return_rewards: return r_iter

    def plot_rewards_over_episodes(self, cumulative=False, fig_size=(8, 8)):
        items = self._get_items()
        r_iter = [(el.reward[0] if np.ndim(el.reward) == 0 else np.average(el.reward), el.episode) for el in items]
        rewards, episodes = zip(*r_iter)
        if cumulative: rewards = np.cumsum(rewards)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        fig.suptitle(f'{self.learn.model.name} for {self.ds.env.spec._env_name}')
        ax.set_ylabel('Rewards' if not cumulative else 'Cumulative Rewards')
        ax.set_xlabel('Episodes')
        ax.xaxis.set_ticks([i for i, el in enumerate(episodes) if episodes[i - 1] != el or i == 0])
        ax.xaxis.set_ticklabels([el for i, el in enumerate(episodes) if episodes[i - 1] != el or i == 0])
        ax.plot(rewards)
        plt.show()

    def episode_video_frames(self, episode=None) -> Dict[str, np.array]:
        """ Returns numpy arrays representing purely episode frames. """
        items = self._get_items(False)
        if episode is None: episode_frames = {key: None for key in list(set([_.episode for _ in items]))}
        else: episode_frames = {episode: None}

        for key in episode_frames:
            if self.ds.feed_type == FEED_TYPE_IMAGE:
                episode_frames[key] = np.array([_.current_state for _ in items if key == _.episode])
            else:
                episode_frames[key] = np.array([_.alternate_state for _ in items if key == _.episode])

        return episode_frames

    def episode_to_gif(self, episode=None, path='', fps=30):
        frames = self.episode_video_frames(episode)

        for ep in frames:
            fig, ax = plt.subplots()
            animation = VideoClip(partial(self._make_frame, frames=frames[ep], axes=ax, fig=fig, title=f'Episode {ep}'),
                                  duration=frames[ep].shape[0])
            animation.write_gif(path + f'episode_{ep}.gif', fps=fps)

    def _make_frame(self, t, frames, axes, fig, title):
        axes.clear()
        fig.suptitle(title)
        axes.imshow(frames[int(t)])
        return mplfig_to_npimage(fig)

    def iplot_episode(self, episode, fps=30):
        if episode is None: raise ValueError('The episode cannot be None for jupyter display')
        x = self.episode_video_frames(episode)[episode]
        fig, ax = plt.subplots()

        self.current_animation = VideoClip(partial(self._make_frame, frames=x, axes=ax, fig=fig,
                                                   title=f'Episode {episode}'), duration=x.shape[0])
        self.current_animation.ipython_display(fps=fps, loop=True, autoplay=True)

    def get_memory_samples(self, batch_size=None, key='reward'):
        samples = self.learn.model.memory.sample(self.learn.model.batch_size if batch_size is None else batch_size)
        if not samples: raise IndexError('Your tree seems empty.')
        if batch_size is not None and batch_size > len(self.learn.model.memory):
            raise IndexError(f'Your batch size {batch_size} > the tree\'s batch size {len(self.learn.model.memory)}')
        if key not in samples[0].obj.keys(): raise ValueError(f'Key {key} not in {samples[0].obj.keys()}')
        return [s.obj[key] for s in samples]

    def plot_memory_samples(self, batch_size=None, key='reward', fig_size=(8, 8)):
        values_of_interest = self.get_memory_samples(batch_size, key)
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()
        fig.suptitle(f'{self.learn.model.name} for {self.ds.env.spec._env_name}')
        ax.set_ylabel(key)
        ax.set_xlabel('Values')
        ax.plot(values_of_interest)
        plt.show()