import io
from itertools import permutations, combinations_with_replacement, product
from PIL import Image
from fastai.train import Interpretation, DatasetType, copy
from gym.spaces import Box
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.linalg import LinAlgError
from torch import Tensor, nn
import numpy as np
from typing import List, Tuple
import torch
import scipy.stats as st
import matplotlib.pyplot as plt
from fast_rl.core import Learner
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice


class AgentInterpretationAlpha(Interpretation):
    def __init__(self, learn: Learner, ds_type: DatasetType=DatasetType.Valid):  # , losses: Tensor):
        """
        Handles converting a learner, and it's runs into useful human interpretable information.

        Notes:
            This class is called AgentInterpretationAlpha because it will overall get deprecated.
            The final working version will be called AgentInterpretation.

        Args:
            learn:
        """
        super().__init__(learn, None, None, None, ds_type=ds_type)

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

    def reward_heatmap(self, episode_slices: List[MarkovDecisionProcessSlice]):
        """
        Takes a state_space and uses the agent to heat map rewards over the space.

        We first need to determine if the state space is discrete or continuous.

        Args:
            state_space:

        Returns:

        """
        current_state_slice = [p for p in product(np.arange(min(self.ds.env.observation_space.low), max(self.ds.env.observation_space.high) + 1), repeat=len(self.ds.env.observation_space.high))]
        heat_map = np.zeros(np.add(self.ds.env.observation_space.high, 1))
        with torch.no_grad():
            for state in current_state_slice:
                heat_map[state] = self.learn.model(torch.from_numpy(np.array(state))).max().numpy()
        return heat_map

    def plot_heatmapped_episode(self, episode):
        """
        Generates plots of heatmapped state spaces for analyzing reward distribution.

        Currently only makes sense for grid based envs. Will be expecting gym_maze environments that are discrete.

        Returns:

        """
        if not str(self.ds.env.spec).__contains__('maze'):
            raise NotImplementedError('Currently only supports gym_maze envs that have discrete state spaces')
        if not isinstance(self.ds.state_size, Box):
            raise NotImplementedError('Currently only supports Box based state spaces with 2 dimensions')

        items = self.ds.x.items  # type: List[MarkovDecisionProcessSlice]
        heat_maps = []

        # For each episode
        buffer = []
        for item in [i for i in items if i.episode == episode]:
            buffer.append(item)
        heat_map = self.reward_heatmap(buffer)
        heat_maps.append((copy(heat_map), copy(buffer[-1]), copy(episode)))

        plots = []
        for single_heatmap in [heat_maps[-1]]:
            fig, ax = plt.subplots(1, 2, figsize=(13, 5))  # type: Tuple[Figure, List[Axes]]
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

        return plots

    def plot_episode(self, episode):
        items = self.ds.x.items  # type: List[MarkovDecisionProcessSlice]

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
            agent_reward_plots = [self.learn.model(torch.from_numpy(np.array(i.current_state))).max().numpy() for i in buffer]
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

        return plots

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
        Heat maps the density of actual vs estimated q values. Good reference for this is at [1].

        References:
            [1] "Simple Example Of 2D Density Plots In Python." Medium. N. p., 2019. Web. 31 Aug. 2019.
            https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67

        Returns:

        """
        items = self.ds.x.items  # type: List[MarkovDecisionProcessSlice]
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
        if episode_num is None: plt.title('2D Gaussian Kernel Q Density Estimation')
        else: plt.title(f'2D Gaussian Kernel Q Density Estimation for episode {episode_num}')
        plt.show()

