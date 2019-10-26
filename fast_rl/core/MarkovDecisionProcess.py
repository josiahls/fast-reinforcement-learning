import gym
from fastai.basic_train import LearnerCallback, DatasetType
from gym.spaces import Discrete, Box, MultiDiscrete

from fast_rl.util.exceptions import MaxEpisodeStepsMissingError

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
from fastai.data_block import ItemList, Tensor, Dataset, DataBunch, DataLoader
from fastai.imports import torch
from datetime import datetime
import pickle

from fast_rl.util.misc import b_colors, list_in_str

FEED_TYPE_IMAGE = 0
FEED_TYPE_STATE = 1


@dataclass
class Bounds(object):
    r"""
    Handles bounds for 1 dimensional spaces.

    between (gym.Space): A convenience variable for determining the min and max bounds

    discrete (bool): Whether the Bounds are discrete or not. Important for N possible values calc.

    min (list): Correlated min for a given dimension

    max (list): Correlated max for a given dimension
    """
    between: Any = None
    discrete: bool = False
    min: list = None
    max: list = None

    def __len__(self):
        r"""
        Returns the number of dimensions the bounds have.

        `self.min`'s length is returned because `self.max` and `self.min` as validated to have the same length.
        """
        return len(self.min)

    @property
    def n_possible_values(self):
        """
        Returns the maximum number of values that can be taken.

        This is important for doing embeddings.
        """
        if not self.discrete: return np.inf
        else: return np.prod(np.subtract(self.max, self.min))

    def __post_init__(self):
        """Sets min and max fields then validates them."""
        self.min, self.max = ifnone(self.min, []), ifnone(self.max, [])
        # If a tuple has been passed, break it into correlated min max variables.
        if self.between is not None:
            for b in (self.between if isinstance(self.between, gym.spaces.Tuple) else listify(self.between)):
                if isinstance(b, (int, np.int64, np.int, float, np.float)):
                    self.min, self.max = self.min + [0], self.max + [b]
                    self.discrete = isinstance(b, (int, np.int, np.int64))
                elif isinstance(b, Discrete):
                    self.min, self.max, self.discrete = self.min + [0], self.max + [b.n], True
                elif isinstance(b, MultiDiscrete):
                    self.min, self.max, self.discrete = self.min + [0] * sum(b.nvec.shape), self.max + b.nvec, True
                elif isinstance(b, Box):
                    self.min, self.max = self.min + list(b.low), self.max + list(b.high)
                    self.discrete = any([b.dtype in (int, np.int, np.int64)])
                else:
                    raise ValueError(f'Tuple not understood {self.between}')

        if len(self.min) != len(self.max):
            raise ValueError(f'Min Max do not match min {len(self.min)} max {len(self.max)}')
        if len(self.min) == 0: raise ValueError(f'Min and Max are 0')


@dataclass
class Action(object):
    r"""
    Handles actions, action space, and value verification.

    An important difference between taken_action and raw_action is that the raw action is the immediate
    raw output from the model before any argmax processing.

    taken_action (np.array): Expected to always to be numpy arrays with shape (-1, 1). This is the action that
    is to be input into the env `step` function.

    raw_action (np.array): Expected to always to be numpy arrays with shape (-1, 1). This is the raw model
    output such as neural net final layer output. Can be None.

    action_space (gym.Space): Used for estimating the max number of values. This is important for embeddings.

    bounds (tuple): Maximum and minimum values for each action dimension.

    n_possible_values (int): An integer or inf value indicating the total number of possible actions there are.
    """
    taken_action: torch.tensor
    action_space: gym.Space
    raw_action: torch.tensor = None
    bounds: Bounds = None
    n_possible_values: int = 0

    def __post_init__(self):
        # Determine bounds
        self.bounds = Bounds(self.action_space)
        self.n_possible_values = self.bounds.n_possible_values

        # Fix shapes
        self.taken_action = torch.tensor(data=self.taken_action).reshape(1, -1)
        if self.raw_action is not None: self.raw_action = torch.tensor(data=self.raw_action).reshape(1, -1)

    def get_single_action(self):
        """ OpenAI envs do not like 1x1 arrays when they are expecting scalars, so we need to unwrap them. """
        a = self.taken_action.detach().numpy()[0]
        if len(self.bounds) == 1: a = a[0]
        if self.bounds.discrete and len(self.bounds) == 1: return int(a)
        elif self.bounds.discrete and len(self.bounds) != 1: return a.astype(int)
        elif not self.bounds.discrete and len(self.bounds) == 1: return [float(a)]
        elif not self.bounds.discrete and len(self.bounds) != 1: return a.reshape(-1,).astype(float)
        raise ValueError(f'This should not have crashed.')

    def set_single_action(self, action: np.array):
        if np.isscalar(action):
            self.taken_action = torch.tensor(data=action).reshape(1, -1)
        elif len(action.shape) == 1:
            self.taken_action = torch.tensor(data=action).reshape(1, -1)
        elif len(action.shape) == 3 and action.shape[0] == 1:
            self.taken_action = torch.tensor(data=action)[0]
        self.taken_action = self.taken_action.int() if self.bounds.discrete else self.taken_action.float()


@dataclass
class State(object):
    r"""
    Handles states, both their main and alternate formats.

    s (np.array): State space acquired from `env.reset` `s_prime` or `env.render`.

    s_prime (np.array): State space acquired from `env.step` or `env.render`.

    alt_s (np.array): Alternate State space acquired from `env.reset` `s_prime` or `env.render`. Should be an image.

    alt_s_prime (np.array): Alternate State space acquired from `env.step` or `env.render`. Should be an image.

    mode (int): Should be either FEED_TYPE_IMAGE or FEED_TYPE_STATE

    observation_space (gym.Space): Used for estimating the max number of values. This is important for embeddings.

    bounds (Bounds): Maximum and minimum values for each state dimension.

    n_possible_values (int): An integer or inf value indicating the total number of possible actions there are.
    """
    s: torch.tensor
    s_prime: torch.tensor
    alt_s: Union[torch.tensor, np.array]
    alt_s_prime: Union[torch.tensor, np.array]
    observation_space: gym.Space
    mode: int = FEED_TYPE_STATE
    bounds: Bounds = None
    n_possible_values: int = 0

    def __str__(self):
        out = copy(self.__dict__)
        for key in out:
            if out[key] is not None and (key == 's' or list_in_str(key, ['_s', 's_'])):  out[key] = out[key].shape

        return f'State: ' + ', '.join([str(i) for i in out.items()])

    def _fix_field(self, input_field):
        if input_field is None: return None
        input_field = copy(input_field)

        if type(input_field) is str: input_field = np.array(input_field)
        elif type(input_field) is tuple:
            dtype = int if self.bounds.discrete else float
            input_field = torch.tensor(data=np.array(input_field).reshape(1, -1).astype(dtype))
        elif np.isscalar(input_field): input_field = torch.tensor(data=input_field)
        elif type(input_field) is torch.Tensor: input_field = input_field.clone().detach()
        else: input_field = torch.from_numpy(input_field)

        # If a non-image state missing the batch dim
        if len(input_field.shape) <= 1: return input_field.reshape(1, -1)
        # If a non-image 2+d state missing the batch dim
        elif input_field.shape[0] != 1 and len(input_field.shape) != 3: return input_field.reshape(1, -1)
        # If an image state and missing the batch dim
        elif input_field.shape[0] != 1 and len(input_field.shape) == 3: return input_field.unsqueeze(0)
        # If an image with 4 dims (b, w, h, c), return safely
        elif input_field.shape[0] == 1 and len(input_field.shape) == 4: return input_field
        # If not an image, but has 2 dims (b, d), return safely
        elif input_field.shape[0] == 1 and len(input_field.shape) == 2: return input_field
        raise ValueError(f'Input has shape {input_field} for mode {self.mode}. This is unexpected.')

    def __post_init__(self):
        if self.mode not in [FEED_TYPE_IMAGE, FEED_TYPE_STATE]:
            ValueError(f'Mode invalid {self.mode} not valid feed type')
        # We want to swap the state variables if the alt is the image state
        if self.mode == FEED_TYPE_IMAGE and len(self.alt_s.shape) > 2 and len(self.s.shape) != 3:
            self.observation_space = gym.spaces.Box(0, 255, self.alt_s.shape, dtype=np.int64)
            self.alt_s, self.alt_s_prime, self.s, self.s_prime = self.s, self.s_prime, self.alt_s, self.alt_s_prime
        # Determine bounds
        self.bounds = Bounds(self.observation_space)
        self.n_possible_values = self.bounds.n_possible_values
        # Fix Shapes
        self.s, self.s_prime = self._fix_field(self.s), self._fix_field(self.s_prime)
        self.alt_s, self.alt_s_prime = self._fix_field(self.alt_s), self._fix_field(self.alt_s_prime)


@dataclass
class MDPStep(object):
    r"""
    Contains all the variables to represent a Markov Decision Process step.

    action (Action):

    state (State):

    done (bool):

    reward (float):

    episode (int):

    step (int):

    """
    action: Action
    state: State
    done: torch.tensor
    reward: torch.tensor
    episode: int
    step: int

    def __post_init__(self):
        self.action = deepcopy(self.action)
        self.state = deepcopy(self.state)
        self.reward = torch.tensor(data=self.reward).reshape(1, -1).float()
        self.done = torch.tensor(data=self.done).reshape(1, -1).float()

    def __str__(self): return ', '.join([str(self.__dict__[el]) for el in self.__dict__])

    def clean(self):
        r""" Removes fields that are generally unimportant (purely debugging) """
        self.state.alt_s_prime = None
        self.state.alt_s = None
        self.state.observation_space = None
        self.state.bounds = None
        self.state.n_possible_values = None
        self.action.raw_action = None
        self.action.action_space = None
        self.action.bounds = None
        self.action.n_possible_values = None

    @property
    def data(self): return self.state.s_prime[0], self.state.alt_s_prime[0]
    @property
    def obj(self): return self.__dict__
    @property
    def s(self): return self.state.s
    @property
    def s_prime(self): return self.state.s_prime
    @property
    def alt_s_prime(self): return self.state.alt_s_prime
    @property
    def a(self): return self.action.taken_action
    @property
    def d(self):
        return bool(self.done)


class MDPCallback(LearnerCallback):
    def __init__(self, learn):
        """
        Handles action assignment, episode naming.

        Args:
            learn:
        """
        super().__init__(learn)
        self.train_ds: MDPDataset = learn.data.train_ds
        self.valid_ds: MDPDataset = None if learn.data.empty_val else learn.data.valid_ds

    def on_batch_begin(self, last_input, last_target, train, **kwargs: Any):
        """ Set the Action of a dataset, determine if still warming up. """
        a = self.learn.predict(last_input)
        if train: self.train_ds.action = Action(taken_action=a, action_space=self.train_ds.action.action_space)
        else: self.valid_ds.action = Action(taken_action=a, action_space=self.train_ds.action.action_space)
        self.train_ds.is_warming_up = self.learn.model.warming_up
        if self.valid_ds is not None: self.valid_ds.is_warming_up = self.learn.model.warming_up
        if not self.learn.model.warming_up and self.learn.loss_func is None:
            self.learn.init_loss_func()
        return {'skip_bwd': True}

    def on_backward_end(self, **kwargs: Any):
        return {'skip_step': True}

    def on_step_end(self, **kwargs: Any):
        return {'skip_zero': True}

    def on_epoch_end(self, last_metrics, epoch, **kwargs: Any) -> None:
        """ Updates the most recent episode number in both datasets. """
        self.train_ds.x.set_recent_run_episode(epoch)
        self.train_ds.episode = epoch
        if last_metrics[0] is not None:
            self.valid_ds.x.set_recent_run_episode(epoch)
            self.valid_ds.episode = epoch


class MDPMemoryManager(LearnerCallback):
    def __init__(self, learn, strategy, k=1):
        super().__init__(learn)
        self.strategy = strategy
        self.k = k
        self._strategy_fn_dict = {
            'k_partitions_top': self.k_top
        }

    def _comp_less(self, key, info, episode):
        return info[key] < info[episode]

    def _comp_greater(self, key, info, episode):
        return info[key] > info[episode]

    def k_top(self, info: Dict[int, List[Tuple[float, bool]]]):
        # If the episode -1 is defined, then clean it, this is just placeholder data
        if -1 in info and not info[-1][1]: return [-1]
        # If the number of not-clean episodes is less than k, don't do anything
        if len([k for k in info if not info[k][1]]) <= self.k: return None
        # Collect k top episodes, then clean the rest
        k_top = []
        for episode in [i for i in info if i != -1]:
            # If the episode is greater than all of the other episodes that are not already in k_top
            compared = [info[episode][0] > info[k][0] for k in info if k not in k_top and k != -1 and k != episode]
            if all(compared) and len(compared) != 0 and len(k_top) < self.k:
                k_top.append(episode)

        return list(set([k for k in info if not info[k][1]]) - set(k_top))


    def on_epoch_end(self, **kwargs: Any):
        for ds_type in [DatasetType.Train] if self.learn.data.empty_val else [DatasetType.Train, DatasetType.Valid]:
            ds: MDPDataset = self.learn.dl(ds_type).dataset
            episodes = self._strategy_fn_dict[self.strategy](ds.x.info)
            if episodes is not None:
                for e in episodes: ds.x.clean(e)


class MDPDataset(Dataset):
    def __init__(self, env: gym.Env, memory_manager, bs, render='rgb_array', feed_type=FEED_TYPE_STATE, max_steps=None):
        r"""
        Handles env execution and ItemList building.

        Args:
            env: OpenAI environment to execute.
            memory_manager: Handles how the list size will be reduced sch as removing image data.
            bs: Size of a single batch for models and the dataset to use.
        """
        self.env = env
        self.render = render
        self.feed_type = feed_type
        self.bs = bs
        self._max_steps = max_steps
        self.action = Action(taken_action=self.env.action_space.sample(), action_space=self.env.action_space)
        self.state = None
        self.s_prime, self.alt_s_prime = None, None
        self.callback = [MDPCallback, memory_manager]
        # Tracking fields
        self.episode = -1
        self.counter = 0
        self.is_warming_up = True

        # FastAI fields
        self.x = MDPList([])
        self.item: Union[MDPStep, None] = None
        self.new(None)

    def aug_steps(self, steps):
        if self.is_warming_up and steps < self.bs: return self.bs
        return steps

    @property
    def max_steps(self):
        if self._max_steps is not None: return self._max_steps
        if hasattr(self.env, '_max_episode_steps'): return getattr(self.env, '_max_episode_steps')
        if self.env.spec.max_episode_steps is not None: return self.env.spec.max_episode_steps

        msg = f'Env {self.env.spec.id} does not have max episode steps. '
        msg += ' Either pass the max steps as a param, or catch this exception then pass a default max step param.'
        raise MaxEpisodeStepsMissingError(msg)

    @property
    def image(self):
        r""" Needed because of blackjack-v0 env >:( """
        try:
            current_image = self.env.render('rgb_array')
            if self.render == 'human': self.env.render(self.render)
        except NotImplementedError:
            print(f'{b_colors.WARNING} {self.env.unwrapped.spec} Not returning Image {b_colors.ENDC}')
            current_image = None
        return current_image

    def __del__(self):
        self.env.close()

    def __len__(self):
        return self.aug_steps(self.max_steps)

    def stage_1_env_reset(self):
        r"""
        Handles environment resetting and dataset batch termination.

        We are interested in the entire dataset ending when an item is done.

        Returns: The state space and the image after a reset.

        """
        if self.counter != 0 and self.item.d:
            self.counter = 0
            if not self.is_warming_up: raise StopIteration
        if self.item is None or self.item.d: return self.env.reset(), self.image
        return self.s_prime, self.alt_s_prime

    def stage_2_env_step(self) -> Tuple[np.array, float, bool, None, np.array]:
        r"""
        Handles taking a step in the environment.

        We want to cancel the env early if we are at our max step amount.

        Returns: The state, reward, whether the episode is done, and the image.
        """
        s_prime, reward, done, _ = self.env.step(self.action.get_single_action())
        # If we are at the max steps limit but the env is not done, we need to force an env end.
        # However, we need the loop to iterate +1 time for allowing the stage_1_env_reset
        if len(self) - 2 == self.counter: done = True
        return s_prime, reward, done, _, self.image

    def new(self, _):
        s, alt_s = self.stage_1_env_reset()
        self.s_prime, reward, done, _, self.alt_s_prime = self.stage_2_env_step()
        # If both the current item and the done are both true, then we need to retry the env
        if self.item is not None and self.item.d and done: return self.new()

        self.state = State(s, self.s_prime, alt_s, self.alt_s_prime,  self.env.observation_space, self.feed_type)
        self.item = MDPStep(self.action, self.state, done, reward, self.episode, self.counter)
        self.counter += 1

        return MDPList([self.item])

    def __getitem__(self, _):
        item = self.new(_)
        self.x.add(item)
        return self.x[-1]

    def to_csv(self, root_path, name):
        if not os.path.exists(root_path): os.makedirs(root_path)
        self.x.to_df().to_csv(root_path / (name + '.csv'), index=False)

    def to_pickle(self, root_path, name):
        if not os.path.exists(root_path): os.makedirs(root_path)
        if not self.x: raise IOError('The dataset is empty, cannot pickle.')
        pickle.dump(self.x, open(root_path / (name + ".pickle"), "wb"), pickle.HIGHEST_PROTOCOL)


class MDPDataBunch(DataBunch):

    def __del__(self):
        if self.train_dl is not None: del self.train_dl.train_ds
        if self.valid_dl is not None: del self.valid_dl.valid_ds

    @property
    def state_action_sample(self) -> Union[Tuple[State, Action], None]:
        ds = ifnone(self.train_ds, self.valid_ds)  # type: MDPDataset
        return ds.state, ds.action if ds is not None else None

    @classmethod
    def from_env(cls, env_name='CartPole-v1', max_steps=None, render='rgb_array', bs: int = 64,
                 feed_type=FEED_TYPE_STATE, num_workers: int = 0, memory_management_strategy='k_partitions_top',
                 split_env_init=True, device: torch.device = None, no_check: bool = False,
                 add_valid=True, **dl_kwargs):

        env = gym.make(env_name)
        memory_manager = partial(MDPMemoryManager, strategy=memory_management_strategy)
        train_list = MDPDataset(env, max_steps=max_steps, feed_type=feed_type, render=render, bs=bs,
                                memory_manager=memory_manager)
        if add_valid:
            valid_list = MDPDataset(env if split_env_init else gym.make(env_name), max_steps=max_steps,
                                    render=render, bs=bs,  feed_type=feed_type, memory_manager=memory_manager)
        else:
            valid_list = None
        path = './data/' + env_name.split('-v')[0].lower() + datetime.now().strftime('%Y%m%d%H%M%S')
        return cls.create(train_list, valid_list, num_workers=num_workers, bs=1, device=device, **dl_kwargs)

    @classmethod
    def from_pickle(cls, env_name='CartPole-v1', bs: int = 1, feed_type=FEED_TYPE_STATE, render='rgb_array',
                    max_steps=None, add_valid=True, num_workers: int = defaults.cpus, path: PathOrStr = None,
                    device: torch.device = None, **dl_kwargs):

        if path is None:
            path = [_ for _ in os.listdir('./data/') if _.__contains__(env_name.split('-v')[0].lower())]
            if not path: raise IOError(f'There is no pickle dirs file found in ./data/ with the env name {env_name}')
            path = Path('./data/' + path[0])

        env = gym.make(env_name)
        train_ls = pickle.load(open(path / 'train.pickle', 'rb'))
        train_list = MDPDataset(env, max_steps=max_steps, render=render, bs=bs, x=train_ls)

        if add_valid:
            valid_ls = pickle.load(open(path / 'valid.pickle', 'rb'))
            valid_list = MDPDataset(env, max_steps=max_steps, render=render, bs=bs, x=valid_ls)
        else:
            valid_list = None

        if path is None: path = './data/' + env_name.split('-v')[0].lower() + datetime.now().strftime('%Y%m%d%H%M%S')

        return cls.create(train_list, valid_list, num_workers=num_workers, path=path, bs=bs, feed_type=feed_type,
                          val_bs=1, device=device, **dl_kwargs)

    @classmethod
    def create(cls, train_ds: MDPDataset, valid_ds: MDPDataset = None, bs: int = 1,
               num_workers: int = defaults.cpus, device: torch.device = None, **dl_kwargs) -> 'DataBunch':
        """Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`.
        Passes `**dl_kwargs` to `DataLoader()`

        Since this is a MarkovProcess, the batches need to be `bs=1` (for now...)
        """
        datasets = cls._init_ds(train_ds, valid_ds, None)
        dls = [DataLoader(d, b, shuffle=s, drop_last=s, num_workers=num_workers, **dl_kwargs) for d, b, s in
               zip(datasets, (bs, bs, bs, bs), (False, False, False, False)) if d is not None]
        databunch = cls(*dls, **dl_kwargs)
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
        return [o for o in (train_ds, valid_ds, copy(train_ds), test_ds) if o is not None]


class MDPList(ItemList):
    _bunch = MDPDataBunch

    def __init__(self, items: Iterator, **kwargs):
        """
        Represents a MDP sequence over episodes.


        Notes:
            Two important fields for you to be aware of: `items` and `x`.
            `x` is just the values being used for directly being feed into the model.
            `items` contains an ndarray of MarkovDecisionProcessSliceAlpha instances. These contain the the primary values
            in x, but also the other important properties of a MDP.

        Args:
            items:
            feed_type:
            **kwargs:
        """
        super().__init__(items, **kwargs)
        self.info = {}

    def set_recent_run_episode(self, episode):
        for i, item in enumerate(reversed(self.items)):
            if item.d and i != 0: break
            item.episode = episode
            self._update_info(episode, item)

    def clean(self, episode):
        self.info[episode][1] = True
        [item.clean() for item in self.items if item.episode == episode]

    def _update_info(self, ep, item: MDPStep):
        self.info[ep] = float(np.sum(self.info[ep][0] + float(item.reward))) if ep in self.info else float(item.reward)
        self.info[ep] = [self.info[ep], False]

    def add(self, items: 'ItemList'):
        [self._update_info(item.episode, item) for item in items.items]
        super().add(items)

    def to_df(self): return pd.DataFrame([i.obj for i in self.items])

    def to_dict(self): return [i.obj for i in self.items]

    def get(self, i): return self.items[i].data

    def reconstruct(self, t: Tensor, x: Tensor = None):
        raise NotImplementedError('Not sure when this will be important.')
