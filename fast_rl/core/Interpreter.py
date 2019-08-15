import io
from itertools import permutations, combinations_with_replacement, product
from PIL import Image
from fastai.train import Interpretation, DatasetType, copy
from gym.spaces import Box
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor, nn
import numpy as np
from typing import List, Tuple
import torch

import matplotlib.pyplot as plt
from fast_rl.core import Learner
from fast_rl.core.Learner import AgentLearner
from fast_rl.core.MarkovDecisionProcess import MarkovDecisionProcessSlice


class AgentInterpretationv1(Interpretation):
    def __init__(self, learn: Learner):  # , losses: Tensor):
        """
        Handles converting a learner, and it's runs into useful human interpretable information.

        Notes:
            This class is called AgentInterpretationv1 because it will overall get deprecated.
            The final working version will be called AgentInterpretation.

        Args:
            learn:
        """
        super().__init__(learn, None, None, None)

    @classmethod
    def from_learner(cls, learn: Learner, ds_type: DatasetType = DatasetType.Valid, activ: nn.Module = None):
        raise NotImplementedError

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

        episode_counter = 0
        # For each episode
        buffer = []
        for item in items:
            buffer.append(item)
            if item.done:
                heat_map = self.reward_heatmap(buffer)
                heat_maps.append((copy(heat_map), copy(item), copy(episode_counter)))
                episode_counter += 1
                buffer = []

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
