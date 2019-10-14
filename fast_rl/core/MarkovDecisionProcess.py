from numbers import Integral

import gym
from fastai.basic_train import LearnerCallback
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
from fastai.data_block import ItemList, Tensor, Dataset, DataBunch, data_collate, DataLoader, PreProcessors
from fastai.imports import torch
from fastai.vision import Image
from gym import error
from gym.envs.algorithmic.algorithmic_env import AlgorithmicEnv
from gym.envs.toy_text import discrete
from gym.wrappers import TimeLimit
from datetime import datetime
import pickle

from fast_rl.util.misc import b_colors

FEED_TYPE_IMAGE = 0
FEED_TYPE_STATE = 1


class MDPMemoryManager(LearnerCallback):
    def __init__(self, learn, mem_strategy, k, max_episodes=None, episode=0, iteration=0):
        """
        Handles different ways of memory management:
        - (k_partitions_best)  keep partition best episodes
        - (k_partitions_worst) keep partition worst episodes
        - (k_partitions_both)  keep partition worst best episodes
        - (k_top_best)         keep high fidelity k top episodes
        - (k_top_worst)        keep k top worst
        - (k_top_both)         keep k top worst and best
        - (none)                keep none, only load into memory (always keep first)
        - (all):               keep all steps will be kept (most memory inefficient)

        Args:
            learn:
            mem_strategy:
            k:
            max_episodes:
            episode:
            iteration:
        """
        super().__init__(learn)
        self.mem_strategy = mem_strategy
        self.k = k
        self.ds = None
        self.max_episodes = max_episodes
        self.episode = episode
        self.last_episode = episode
        self.iteration = iteration
        self._persist = max_episodes is not None

        # Private Util Fields
        self._train_episodes_keep = {}
        self._valid_episodes_keep = {}
        self._current_partition = 0

        self._lower = 0
        self._upper = 0

    def _comp_less(self, key, d, episode): return d[key] < self.data.x.info[episode]
    def _comp_greater(self, key, d, episode): return d[key] > self.data.x.info[episode]
    def _dict_access(self, key, dictionary): return {key: dictionary[key]}

    def _k_top_best(self, episode, episodes_keep):
        best = dict(filter(partial(self._comp_greater, d=episodes_keep, episode=episode), episodes_keep))
        return list(dict(sorted(best.items(), reverse=True)).keys())

    def _k_top_worst(self, episode, episodes_keep):
        worst = dict(filter(partial(self._comp_less, d=episodes_keep, episode=episode), episodes_keep))
        return list(dict(sorted(worst.items())).keys())

    def _k_top_both(self, episode, episodes_keep):
        best, worst = self._k_top_best(episode, episodes_keep), self._k_top_worst(episode, episodes_keep)
        if best: return best
        return worst

    def _k_partitions_best(self, episode, episodes_keep):
        # Filter episodes by their partition
        if episode >= self._upper: self.lower, self._upper = self._upper, self._upper + self.max_episodes // self.k
        filtered_episodes = {ep: episodes_keep[ep] for ep in episodes_keep if self._lower <= ep < self._upper}
        best = list(filter(partial(self._comp_greater, d=filtered_episodes, episode=episode), filtered_episodes))
        best = {k: filtered_episodes[k] for k in filtered_episodes if k in best}
        return list(dict(sorted(best.items(), reverse=True)).keys())

    def _k_partitions_worst(self, episode, episodes_keep):
        # Filter episodes by their partition
        if episode >= self._upper: self.lower, self._upper = self._upper, self._upper + self.max_episodes // self.k
        filtered_episodes = {ep: episodes_keep[ep] for ep in episodes_keep if self._lower <= ep < self._upper}
        worst = list(filter(partial(self._comp_less, d=filtered_episodes, episode=episode), filtered_episodes))
        worst = {k: filtered_episodes[k] for k in filtered_episodes if k in worst}
        return list(dict(sorted(worst.items())).keys())

    def _k_partitions_both(self, episode, episodes_keep):
        best, worst = self._k_partitions_best(episode, episodes_keep), self._k_partitions_worst(episode, episodes_keep)
        if best: return best
        return worst

    def on_loss_begin(self, **kwargs: Any):
        self.iteration += 1

    def manage_memory(self, episodes_keep, episode):
        if episode not in self.ds.x.info: return
        episodes_keep[episode] = self.ds.x.info[episode]
        if len(episodes_keep) < self.k: return
        # We operate of the second to most recent episode so the agent still has an opportunity to use the full episode
        episode = episode - 1
        if episode not in self.ds.x.info: return

        try:
            # If the episodes to keep is full, then we have to decide which ones to remove
            if self.mem_strategy == 'k_top_best': del_ep = self._k_top_best(episode, episodes_keep)
            if self.mem_strategy == 'k_top_worst': del_ep = self._k_top_worst(episode, episodes_keep)
            if self.mem_strategy == 'k_top_both': del_ep = self._k_top_both(episode, episodes_keep)
            if self.mem_strategy == 'k_partitions_best': del_ep = self._k_partitions_best(episode, episodes_keep)
            if self.mem_strategy == 'k_partitions_worst': del_ep = self.k_partitions_worst(episode, episodes_keep)
            if self.mem_strategy == 'k_partitions_both': del_ep = self._k_partitions_both(episode, episodes_keep)
            if self.mem_strategy == 'all': del_ep = [-1]

            # If there are episodes to delete, then set them as the main episode to delete
            if len(del_ep) != 0:
                # episodes_keep[episode] = self.ds.x.info[episode]
                del episodes_keep[del_ep[0]]
                episode = del_ep[0]
                self.ds.x.clean(episode)
            self.ds.x.info = episodes_keep

        except KeyError as e:
            pass
        except TypeError as e:
            pass

    def on_train_begin(self, n_epochs, **kwargs: Any):
        self.max_episodes = n_epochs if not self._persist else self.max_episodes
        self._upper = self.max_episodes // self.k

    def on_epoch_begin(self, epoch, **kwargs: Any):
        self.episode = epoch if not self._persist else self.episode + 1
        self.iteration = 0

    def on_epoch_end(self, **kwargs: Any) -> None:
        if self.learn.data.train_ds is not None:
            self.ds = self.learn.data.train_ds
            self.manage_memory(self._train_episodes_keep, self.episode)
        if self.learn.data.valid_dl is not None:
            self.ds = self.learn.data.valid_ds
            self.manage_memory(self._valid_episodes_keep, self.episode)


class MDPDataset(Dataset):
    def __init__(self, env: gym.Env, feed_type=FEED_TYPE_STATE, render='rgb_array', max_steps=None, bs=8,
                 x=None, memory_management_strategy='k_partitions_best', k=1, skip=False, embeddable=False):
        """
        Handles the running and loading of environments, as well as storing episode steps.

        mem_strategy has a few settings. This is for inference on the existing data:
        - (k_partitions_best) keep partition best episodes
        - (k_partitions_both) keep partition worst best episodes
        - (k_top_best)        keep high fidelity k top episodes
        - (k_top_worst)       keep k top worst
        - (k_top_both)        keep k top worst and best
        - (non)               keep non, only load into memory (always keep first)
        - (all):              keep all steps will be kept (most memory inefficient)

        Args:
            env:
            feed_type:
            render:
            max_steps:
            bs:
            x:
            memory_management_strategy: Reference above. Tests `self` how to keep memory usage down.
            k: Related to the `mem_strategy` and is either k quartiles and k most.
            save_every: Skip adding to datasets.
        """
        self.k = k
        self.skip = False
        if skip:
            self.skip = skip
            return
        self.mem_strat = memory_management_strategy
        self.bs = bs
        # noinspection PyUnresolvedReferences,PyProtectedMember
        env._max_episode_steps = env.spec.max_episode_steps if not hasattr(env, '_max_episode_steps') else env._max_episode_steps
        self.max_steps = env._max_episode_steps if max_steps is None else max_steps
        self.render = render
        self.feed_type = feed_type
        self.env = env
        # MDP specific values
        self.actions = self.get_random_action(env.action_space)
        if isinstance(env.action_space, Box): self.raw_action = np.random.randn((env.action_space.shape[0]))
        elif isinstance(env.action_space, Discrete): self.raw_action = np.random.randn((env.action_space.n))
        else: self.raw_action = self.get_random_action(env.action_space)

        self.is_done = True
        self.current_state = None
        self.current_image = None

        self.embeddable = embeddable

        self.env_specific_handle()
        self.counter = -1
        self.episode = 0
        self.x = MarkovDecisionProcessList() if x is None else x  # self.new(0)
        self.item = None
        self.episodes_to_keep = {}

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
            current_image = self.env.render('rgb_array')
            if self.render == 'human': self.env.render(self.render)
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
        if self.is_done or self.counter >= self.max_steps - 3:
            self.current_state, reward, self.is_done, info = self.env.reset(), 0, False, {}
            if type(self.current_state) is not list and type(self.current_state) is not np.ndarray: self.current_state = [self.current_state]
            # Specifically for the stupid blackjack-v0 env >:(
            self.current_image = self._get_image()

        result_state, reward, self.is_done, info = self.env.step(self.actions)
        if self.is_done and self.counter == -1:
            self.current_state, reward, self.is_done, info = self.env.reset(), 0, False, {}

        if type(result_state) is not list and type(result_state) is not np.ndarray: result_state = [result_state]
        result_image = self._get_image()
        self.counter += 1

        # Second Phase: Generate MDP slice
        result_state = result_image.transpose(2, 0, 1) if self.feed_type == FEED_TYPE_IMAGE and result_image is not None else result_state
        current_state = self.current_image.transpose(2, 0, 1) if self.feed_type == FEED_TYPE_IMAGE and self.current_image is not None else self.current_state
        alternate_state = result_state if self.feed_type == FEED_TYPE_IMAGE or result_state is None else result_image
        items = MarkovDecisionProcessSlice(state=np.copy(current_state), state_prime=np.copy(result_state),
                                           alt_state=np.copy(alternate_state), action=np.copy(self.actions),
                                           reward=reward, done=copy(self.is_done), feed_type=copy(self.feed_type),
                                           episode=copy(self.episode), raw_action=self.raw_action)
        self.current_state = copy(result_state)
        self.current_image = copy(result_image)

        list_item = MarkovDecisionProcessList([items])
        return list_item

    def __len__(self):
        return self.max_steps

    def __getitem__(self, _) -> 'MDPDataset':
        item = self.new(_)
        if (self.x and self.is_done and self.counter != -1) or \
                (self.counter >= self.max_steps - 2):
            self.x.add(item)
            self.counter = -1
            self.episode += 1
            raise StopIteration

        self.x.add(item)
        # x = self.x[idxs]  # Perhaps have this as an option?
        x = self.x[-1]
        return x.copy() if isinstance(x, np.ndarray) else x

    def to_csv(self, root_path, name):
        df = self.x.to_df()  # type: pd.DateFrame()
        if not os.path.exists(root_path): os.makedirs(root_path)
        df.to_csv(root_path / (name + '.csv'), index=False)

    def to_pickle(self, root_path, name):
        if not os.path.exists(root_path): os.makedirs(root_path)
        if not self.x: raise IOError('The dataset is empty, cannot pickle.')
        pickle.dump(self.x, open(root_path / (name + ".pickle"), "wb"), pickle.HIGHEST_PROTOCOL)


class MDPDataBunch(DataBunch):
    def _get_sizes_and_possible_values(self, item):
        if isinstance(item, Discrete) and len(item.shape) != 0: return item.n, item.n
        if isinstance(item, Discrete) and len(item.shape) == 0: return 1, item.n
        if isinstance(item, Box) and (item.dtype == int or item.dtype == np.uint8):
            return item.shape if len(item.shape) > 1 else item.shape[0], np.prod(item.high)
        if isinstance(item, Box) and item.dtype == np.float32:
            return item.shape if len(item.shape) > 1 else item.shape[0], np.inf

    # noinspection PyUnresolvedReferences
    def get_action_state_size(self):
        if self.train_ds is not None:
            a_s, s_s = self.train_ds.env.action_space, self.train_ds.state_size
        elif self.valid_ds is not None:
            a_s, s_s = self.valid_ds.env.action_space, self.valid_ds.state_size
        else:
            return None
        return tuple(map(self._get_sizes_and_possible_values, [a_s, s_s]))

    @classmethod
    def from_env(cls, env_name='CartPole-v1', max_steps=None, render='rgb_array', test_ds: Optional[Dataset] = None,
                 path: PathOrStr = None, bs: int = 1, feed_type=FEED_TYPE_STATE, val_bs: int = None,
                 num_workers: int = 0, embed=False, memory_management_strategy='k_partitions_both',
                 dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
                 collate_fn: Callable = data_collate, no_check: bool = False, add_valid=True, **dl_kwargs):
        """


        Args:
            env_name:
            max_steps:
            render:
            test_ds:
            path:
            bs:
            feed_type:
            val_bs:
            num_workers:
            embed:
            memory_management_strategy: has a few settings. This is for inference on the existing data.
                - (k_partitions_best) keep partition best episodes
                - (k_partitions_both) keep partition worst best episodes
                - (k_top_best)        keep high fidelity k top episodes
                - (k_top_worst)       keep k top worst
                - (k_top_both)        keep k top worst and best
                - (non)               keep non, only load into memory (always keep first)
                - (all):              keep all steps will be kept (most memory inefficient)
            dl_tfms:
            device:
            collate_fn:
            no_check:
            add_valid:
            **dl_kwargs:

        Returns:

        """

        try:
            # train_list = MDPDataset(gym.make(env_name), max_steps=max_steps, render=render)
            # valid_list = MDPDataset(gym.make(env_name), max_steps=max_steps, render=render)
            env = gym.make(env_name)
            val_bs = bs if val_bs is None else val_bs
            train_list = MDPDataset(env, max_steps=max_steps, render=render, bs=bs, embeddable=embed,
                                    memory_management_strategy=memory_management_strategy)
            if add_valid: valid_list = MDPDataset(env, max_steps=max_steps, render=render, bs=val_bs, embeddable=embed,
                                                  memory_management_strategy=memory_management_strategy)
            else: valid_list = None
        except error.DependencyNotInstalled as e:
            print('Mujoco is not installed. Returning None')
            if e.args[0].lower().__contains__('mujoco'): return None

        bs, val_bs = 1, None
        path = './data/' + env_name.split('-v')[0].lower() + datetime.now().strftime('%Y%m%d%H%M%S')

        return cls.create(train_list, valid_list, num_workers=num_workers, test_ds=test_ds, path=path, bs=bs,
                          feed_type=feed_type, val_bs=val_bs, dl_tfms=dl_tfms, device=device, collate_fn=collate_fn,
                          no_check=no_check, **dl_kwargs)

    @classmethod
    def from_pickle(cls, env_name='CartPole-v1', max_steps=None, render='rgb_array', test_ds: Optional[Dataset] = None,
                    path: PathOrStr = None, bs: int = 1, feed_type=FEED_TYPE_STATE, val_bs: int = None,
                    num_workers: int = 0,
                    dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
                    collate_fn: Callable = data_collate, no_check: bool = False, add_valid=True, **dl_kwargs):

        if path is None:
            path = [_ for _ in os.listdir('./data/') if _.__contains__(env_name.split('-v')[0].lower())]
            if not path: raise IOError(f'There is no pickle dirs file found in ./data/ with the env name {env_name}')
            path = Path('./data/' + path[0])

        try:
            env = gym.make(env_name)
            val_bs = bs if val_bs is None else val_bs
            train_ls = pickle.load(open(path / 'train.pickle', 'rb'))
            train_list = MDPDataset(env, max_steps=max_steps, render=render, bs=bs, x=train_ls)

            if add_valid:
                valid_ls = pickle.load(open(path / 'valid.pickle', 'rb'))
                valid_list = MDPDataset(env, max_steps=max_steps, render=render, bs=val_bs, x=valid_ls)
            else:
                valid_list = None

        except error.DependencyNotInstalled as e:
            print('Mujoco is not installed. Returning None')
            if e.args[0].lower().__contains__('mujoco'): return None

        bs, val_bs = 1, None
        if path is None: path = './data/' + env_name.split('-v')[0].lower() + datetime.now().strftime('%Y%m%d%H%M%S')

        return cls.create(train_list, valid_list, num_workers=num_workers, test_ds=test_ds, path=path, bs=bs,
                          feed_type=feed_type, val_bs=val_bs, dl_tfms=dl_tfms, device=device, collate_fn=collate_fn,
                          no_check=no_check, **dl_kwargs)

    @classmethod
    def from_csv(cls, env_name='CartPole-v1', max_steps=None, render='rgb_array', test_ds: Optional[Dataset] = None,
                 path: PathOrStr = None, bs: int = 1, feed_type=FEED_TYPE_STATE, val_bs: int = None,
                 num_workers: int = 0,
                 dl_tfms: Optional[Collection[Callable]] = None, device: torch.device = None,
                 collate_fn: Callable = data_collate, no_check: bool = False, add_valid=True, **dl_kwargs):
        raise NotImplementedError('Not implemented for now. Saving state data into a csv seems extremely clunky.'
                                  ' Suggested to use to_pickle and from_pickle due to easier numpy conversion.')

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
        databunch = cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
        if valid_ds is None: databunch.valid_dl = None
        return databunch

    def to_csv(self):
        if self.train_ds is not None: self.train_ds.to_csv(self.path, 'train')
        if self.valid_ds is not None: self.valid_ds.to_csv(self.path, 'valid')

    def to_pickle(self):
        if self.train_ds is not None: self.train_ds.to_pickle(self.path, 'train')
        if self.valid_ds is not None: self.valid_ds.to_pickle(self.path, 'valid')

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
        self.info = {}

    def clean(self, episode):
        for item in self.items:
            if item.episode == episode: item.clean()

    def add(self, items: 'ItemList'):
        # Update the episode related composition information
        for item in items.items:
            if item.episode in self.info: self.info[item.episode] = float(np.sum(self.info[item.episode] + item.reward))
            else: self.info[item.episode] = float(item.reward)

        super().add(items)

    def to_df(self):
        return pd.DataFrame([i.obj for i in self.items])

    def to_dict(self):
        return [i.obj for i in self.items]

    def get(self, i):
        return self.items[i].data

    def reconstruct(self, t: Tensor, x: Tensor = None):
        if self.feed_type == FEED_TYPE_IMAGE:
            return MarkovDecisionProcessSlice(state=Image(t), state_prime=Image(x[0]),
                                              alt_state=Floats(x[1]), action=Floats(x[1]),
                                              reward=Floats(x[2]), done=x[3], feed_type=self.feed_type)
        else:
            return MarkovDecisionProcessSlice(state=Floats(t), state_prime=Floats(x[0]),
                                              alt_state=Image(x[1]), action=Floats(x[1]),
                                              reward=Floats(x[2]), done=x[3], feed_type=self.feed_type)


class MarkovDecisionProcessSlice(ItemBase):
    # noinspection PyMissingConstructor
    def __init__(self, state, state_prime, alt_state, action, reward, done, episode, raw_action,
                 feed_type=FEED_TYPE_IMAGE):
        action = np.copy(action)
        raw_action = np.copy(raw_action)
        if len(action.shape) == 0: action = np.array(action, ndmin=1)
        if isinstance(np.copy(action), int): action = np.array(action, ndmin=1)
        if isinstance(reward, float) or isinstance(reward, int): reward = np.array(reward, ndmin=1)
        self.current_state, self.result_state, self.alternate_state, self.actions, self.reward, self.done, self.episode, self.raw_action = state, state_prime, alt_state, action, reward, done, episode, raw_action
        self.data, self.obj = alt_state if feed_type == FEED_TYPE_IMAGE else state, \
                              {'state': self.current_state, 'state_prime': self.result_state,
                               'alt_state': self.alternate_state, 'action': action, 'reward': reward, 'done': done,
                               'episode': episode, 'feed_type': feed_type, 'raw_action': raw_action}

    def clean(self, only_alt=False):
        if not only_alt:
            self.current_state, self.result_state = None, None
            self.obj['state'], self.obj['state_prime'] = None, None

        self.alternate_state, self.obj['alt_state'] = None, None

    def __str__(self):
        formatted = (
            map(lambda y: f'{y}:{self.obj[y].shape}', filter(lambda y: y.__contains__('state'), self.obj.keys())),
            map(lambda y: f'{y}:{self.obj[y]}', filter(lambda y: not y.__contains__('state'), self.obj.keys()))
        )

        return ', '.join(list(formatted[0]) + list(formatted[1]))

    def to_one(self):
        return Image(self.alternate_state)
