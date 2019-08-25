from numbers import Integral

import gym
from gym.spaces import Discrete, Box

try:
    # noinspection PyUnresolvedReferences
    import pybulletgym.envs

except ModuleNotFoundError as e:
    print(f'Can\'t import one of these: {e}')
try:
    # noinspection PyUnresolvedReferences
    import gym_maze
except ModuleNotFoundError as e:
    print(f'Can\'t import one of these: {e}')

from fastai.core import *
# noinspection PyProtectedMember
from fastai.data_block import ItemList, Tensor, Dataset, DataBunch, data_collate, DataLoader
from fastai.imports import torch
from fastai.vision import Image
from gym import error
from gym.envs.algorithmic.algorithmic_env import AlgorithmicEnv
from gym.envs.toy_text import discrete
from gym.wrappers import TimeLimit

from fast_rl.util.misc import b_colors

FEED_TYPE_IMAGE = 0
FEED_TYPE_STATE = 1


class MDPDataset(Dataset):
    def __init__(self, env: gym.Env, feed_type=FEED_TYPE_STATE, render='rgb_array', max_steps=None):
        # noinspection PyUnresolvedReferences,PyProtectedMember
        self.max_steps = env._max_episode_steps if max_steps is None else max_steps
        self.render = render
        self.feed_type = feed_type
        self.env = env
        # MDP specific values
        self.actions = self.get_random_action(env.action_space)
        self.is_done = True
        self.current_state = None
        self.current_image = None

        self.env_specific_handle()
        self.counter = -1
        self.episode = 0
        self.x = MarkovDecisionProcessList()#self.new(0)
        self.item = None

    @property
    def state_size(self):
        if self.feed_type == FEED_TYPE_STATE:
            return self.env.observation_space
        else:
            return gym.spaces.Box(0, 255, shape=self.env.render('rgb_array').shape)

    def __del__(self):
        self.env.close()

    def env_specific_handle(self):
        if isinstance(self.env, TimeLimit) and isinstance(self.env.unwrapped, AlgorithmicEnv):
            self.render = 'ansi' if self.render == 'rgb_array' else self.render
        if isinstance(self.env, TimeLimit) and isinstance(self.env.unwrapped, discrete.DiscreteEnv):
            self.render = 'ansi' if self.render == 'rgb_array' else self.render

    def get_random_action(self, action_space=None):
        action_space = action_space if action_space is not None else self.env.action_space
        return action_space.sample()

    def _get_image(self):
        # Specifically for the stupid blackjack-v0 env >:(
        try:
            current_image = self.env.render(self.render)
        except NotImplementedError:
            print(f'{b_colors.WARNING} {self.env.unwrapped.spec} Not returning Image {b_colors.ENDC}')
            current_image = None
        return current_image

    def new(self, _):
        """
        New element is a query of the environment

        Args:
            _:

        Returns:

        """
        # First Phase: decide on episode reset. Collect current state and image representations.
        if self.is_done or self.counter >= self.max_steps - 2:
            self.current_state, reward, self.is_done, info = self.env.reset(), 0, False, {}
            # Specifically for the stupid blackjack-v0 env >:(
            self.current_image = self._get_image()

        result_state, reward, self.is_done, info = self.env.step(self.actions)
        result_image = self._get_image()
        self.counter += 1

        # Second Phase: Generate MDP slice
        result_state = result_image if self.feed_type == FEED_TYPE_IMAGE and result_image is not None else result_state
        current_state = self.current_image if self.feed_type == FEED_TYPE_IMAGE and self.current_image is not None else self.current_state
        alternate_state = result_state if self.feed_type == FEED_TYPE_IMAGE or result_state is None else result_image
        items = MarkovDecisionProcessSlice(current_state=np.copy(current_state), result_state=np.copy(result_state),
                                           alternate_state=np.copy(alternate_state), actions=np.copy(self.actions),
                                           reward=reward, done=copy(self.is_done), feed_type=copy(self.feed_type),
                                           episode=copy(self.episode))
        self.current_state = copy(result_state)
        self.current_image = copy(result_image)

        list_item = MarkovDecisionProcessList([items])
        return list_item

    def __len__(self):
        return self.max_steps

    def __getitem__(self, _) -> 'MDPDataset':
        if (self.x and self.is_done and self.counter != -1) or \
                (self.counter >= self.max_steps - 2):
            self.counter = -1
            self.episode += 1
            raise StopIteration
        item = self.new(_)
        self.x.add(item)
        # x = self.x[idxs]  # Perhaps have this as an option?
        x = self.x[-1]
        return x.copy() if isinstance(x, np.ndarray) else x


class MDPDataBunch(DataBunch):
    def _get_sizes(self, item):
        if isinstance(item, Discrete): return item.n
        if isinstance(item, Box) and len(item.shape) == 1: return item.shape[0]

    # noinspection PyUnresolvedReferences
    def get_action_state_size(self):
        if self.train_ds is not None:
            a_s, s_s = self.train_ds.env.action_space, self.train_ds.state_size
        elif self.valid_ds is not None:
            a_s, s_s = self.valid_ds.env.action_space, self.valid_ds.state_size
        else:
            return None
        return tuple(map(self._get_sizes, [a_s, s_s]))

    @classmethod
    def from_env(cls, env_name='CartPole-v1', max_steps=None, render='rgb_array', test_ds: Optional[Dataset] = None,
                 path: PathOrStr = '.', bs: int = 1, feed_type=FEED_TYPE_STATE, val_bs: int = None,
                 num_workers: int = 0,
                 dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
                 collate_fn: Callable = data_collate, no_check: bool = False, **dl_kwargs):

        try:
            # train_list = MDPDataset(gym.make(env_name), max_steps=max_steps, render=render)
            # valid_list = MDPDataset(gym.make(env_name), max_steps=max_steps, render=render)
            env = gym.make(env_name)
            train_list = MDPDataset(env, max_steps=max_steps, render=render)
            valid_list = MDPDataset(env, max_steps=max_steps, render=render)
        except error.DependencyNotInstalled as e:
            print('Mujoco is not installed. Returning None')
            if e.args[0].lower().__contains__('mujoco'): return None

        return cls.create(train_list, valid_list, num_workers=num_workers, test_ds=test_ds, path=path, bs=bs,
                          feed_type=feed_type, val_bs=val_bs, dl_tfms=dl_tfms, device=device, collate_fn=collate_fn,
                          no_check=no_check, **dl_kwargs)

    @classmethod
    def create(cls, train_ds: MDPDataset, valid_ds: MDPDataset = None,
               test_ds: Optional[Dataset] = None, path: PathOrStr = '.', bs: int = 1,
               feed_type=FEED_TYPE_STATE,
               val_bs: int = None, num_workers: int = defaults.cpus, dl_tfms: Optional[Collection[Callable]] = None,
               device: torch.device = None, collate_fn: Callable = data_collate, no_check: bool = False,
               **dl_kwargs) -> 'DataBunch':
        """Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`.
        Passes `**dl_kwargs` to `DataLoader()`

        Since this is a MarkovProcess, the batches need to be `bs=1` (for now...)
        """
        train_ds.feed_type = feed_type
        if valid_ds is not None: valid_ds.feed_type = feed_type
        if test_ds is not None: test_ds.feed_type = feed_type

        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        dls = [DataLoader(d, b, shuffle=s, drop_last=s, num_workers=num_workers, **dl_kwargs) for d, b, s in
               zip(datasets, (bs, val_bs, val_bs, val_bs), (False, False, False, False)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

    @staticmethod
    def _init_ds(train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset] = None):
        # train_ds, but without training tfms
        fix_ds = copy(train_ds)  # valid_ds.new(train_ds.x) if hasattr(valid_ds, 'new') else train_ds
        return [o for o in (train_ds, valid_ds, fix_ds, test_ds) if o is not None]


class MarkovDecisionProcessList(ItemList):
    _bunch = MDPDataBunch

    def __init__(self, items=np.array([]), feed_type=FEED_TYPE_IMAGE, **kwargs):
        """
        Represents a MDP sequence over episodes.


        Notes:
            Two important fields for you to be aware of: `items` and `x`.
            `x` is just the values being used for directly being feed into the model.
            `items` contains an ndarray of MarkovDecisionProcessSlice instances. These contain the the primary values
            in x, but also the other important properties of a MDP.

        Args:
            items:
            feed_type:
            **kwargs:
        """
        super(MarkovDecisionProcessList, self).__init__(items, **kwargs)
        self.feed_type = feed_type
        self.copy_new.append('feed_type')
        self.ignore_empty = True

    def get(self, i):
        res = super(MarkovDecisionProcessList, self).get(i)
        return res.current_state

    def reconstruct(self, t: Tensor, x: Tensor = None):
        if self.feed_type == FEED_TYPE_IMAGE:
            return MarkovDecisionProcessSlice(current_state=Image(t), result_state=Image(x[0]),
                                              alternate_state=Floats(x[1]), actions=Floats(x[1]),
                                              reward=Floats(x[2]), done=x[3], feed_type=self.feed_type)
        else:
            return MarkovDecisionProcessSlice(current_state=Floats(t), result_state=Floats(x[0]),
                                              alternate_state=Image(x[1]), actions=Floats(x[1]),
                                              reward=Floats(x[2]), done=x[3], feed_type=self.feed_type)


class MarkovDecisionProcessSlice(ItemBase):
    # noinspection PyMissingConstructor
    def __init__(self, current_state, result_state, alternate_state, actions, reward, done, episode, feed_type=FEED_TYPE_IMAGE):
        actions = np.copy(actions)
        if len(actions.shape) == 0: actions = np.array(actions, ndmin=1)
        if isinstance(np.copy(actions), int): actions = np.array(actions, ndmin=1)
        if isinstance(reward, float) or isinstance(reward, int): reward = np.array(reward, ndmin=1)
        self.current_state, self.result_state, self.alternate_state, self.actions, self.reward, self.done, self.episode = current_state, result_state, alternate_state, actions, reward, done, episode
        self.data, self.obj = [alternate_state] if feed_type == FEED_TYPE_IMAGE else [current_state], (
            result_state, alternate_state, actions, reward, done, episode)

    def __str__(self):
        return Image(self.alternate_state)

    def to_one(self):
        return Image(self.alternate_state)
