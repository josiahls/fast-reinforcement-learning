from numbers import Integral

import gym
from fastai.core import *
# noinspection PyProtectedMember
from fastai.data_block import ItemList, Tensor, Dataset, DataBunch, ItemLists, data_collate, DataLoader, try_int
from fastai.imports import torch
from fastai.vision import Image


class MarkovDecisionProcessDataset(Dataset):
    def __init__(self, env: gym.Env, feed_type='state', render='rgb_array', max_steps = None):
        self.max_steps = env._max_episode_steps if max_steps is None else max_steps
        self.render = render
        self.feed_type = feed_type
        self.env = env
        self.actions = env.action_space.sample()
        self.is_done = True
        self.reward = None
        self.last_state = None
        self.x = self.new(0)
        self.item = None

    def new(self, index):
        if self.is_done:
            output, self.reward, self.is_done, info = self.env.reset(), 0, False, {}
        else:
            output, self.reward, self.is_done, info = self.env.step(self.actions)

        image = self.env.render(self.render)
        current_state = image if self.feed_type == 'image' else output
        alternate_state = output if self.feed_type == 'image' else image
        items = MarkovDecisionProcessSlice(current_state=current_state, last_state=self.last_state,
                                           alternate_state=alternate_state, actions=self.actions,
                                           reward=self.reward, done=self.is_done, feed_type=self.feed_type)

        return MarkovDecisionProcessList([items])

    def __len__(self):
        return self.max_steps

    def __getitem__(self, idxs: Union[int, np.ndarray])->'MarkovDecisionProcessDataset':
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None:
                self.x.add(self.new(idxs))
                x = self.x[idxs]
            else:
                x = self.item

            return x
        return self.new(idxs)


class MarkovDecisionProcessDataBunch(DataBunch):
    @classmethod
    def create(cls, train_ds: MarkovDecisionProcessDataset, valid_ds: MarkovDecisionProcessDataset,
               test_ds: Optional[Dataset] = None, path: PathOrStr = '.', bs: int = 1,
               feed_type='state',
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
    def _init_ds(train_ds: Dataset, valid_ds: Dataset, test_ds: Optional[Dataset]=None):
        # train_ds, but without training tfms
        fix_ds = valid_ds.new(train_ds.x) if hasattr(valid_ds,'new') else train_ds
        return [o for o in (train_ds,valid_ds,fix_ds,test_ds) if o is not None]


class MarkovDecisionProcessList(ItemList):
    _bunch = MarkovDecisionProcessDataBunch

    def __init__(self, items=np.array([]), feed_type='image', **kwargs):
        super(MarkovDecisionProcessList, self).__init__(items, **kwargs)
        self.feed_type = feed_type
        self.copy_new.append('feed_type')
        self.ignore_empty = True

    def get(self, i):
        res = super(MarkovDecisionProcessList, self).get(i)
        return res.current_state

    def reconstruct(self, t: Tensor, x: Tensor = None):
        if self.feed_type == 'image':
            return MarkovDecisionProcessSlice(current_state=Image(t), last_state=Image(x[0]),
                                              alternate_state=Floats(x[1]), actions=Floats(x[1]),
                                              reward=Floats(x[2]), done=x[3], feed_type=self.feed_type)
        else:
            return MarkovDecisionProcessSlice(current_state=Floats(t), last_state=Floats(x[0]),
                                              alternate_state=Image(x[1]), actions=Floats(x[1]),
                                              reward=Floats(x[2]), done=x[3], feed_type=self.feed_type)


class MarkovDecisionProcessSlice(ItemBase):
    # noinspection PyMissingConstructor
    def __init__(self, current_state, last_state, alternate_state, actions, reward, done, feed_type='image'):
        self.current_state, self.last_state, self.alternate_state, self.actions, self.reward, self.done = current_state, last_state, alternate_state, actions, reward, done
        self.data, self.obj = [alternate_state] if feed_type == 'image' else [current_state], (last_state, alternate_state, actions, reward, done)

    def __str__(self):
        return Image(self.alternate_state)

    def to_one(self):
        return Image(self.alternate_state)
