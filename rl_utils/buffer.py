import numpy as np
import torch as th
import random

from typing import Union, List


def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
    """
    Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
    to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
    to [n_steps * n_envs, ...] (which maintain the order)
    :param arr:
    :return:
    """
    shape = arr.shape
    if len(shape) < 3:
        shape = shape + (1,)
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class ReplayBuffer(object):
    def __init__(self, size: int, device='cpu', vectorized=True):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """

        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._device = device
        self.vectorized = vectorized

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, dev):
        self._device = dev

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def _add(self, obs_t, action, reward, obs_tp1, done, info):
        data = (obs_t, action, reward, obs_tp1, done, info)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def add(self, obs_t, action, reward, obs_tp1, done, info):
        """
        add a new transition to the buffer
        :param obs_t: (Union[np.ndarray, int]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int]) the current observation
        :param done: (bool) is the episode done

        """
        data = (obs_t, action, reward, obs_tp1, done, info)
        if self.vectorized:
            for o, a, r, o_tp1, done, info in zip(*data):
                if done and "is_success" in info.keys():
                    i = float(info["is_success"])
                else:
                    i = 0.
                self._add(o, a, r, o_tp1, done, i)
        else:
            if done and 'is_success' in info.keys():
                i = float(info["is_success"])
            else:
                i = 0.
            self._add(*data[:-1], info=i)

    def extend(self, obs_t, action, reward, obs_tp1, done, info):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done, info):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], ):
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            infos.append(info)
        ret = [obses_t, rewards, obses_tp1, dones]
        ret = [np.asarray(r, dtype=np.float32) for r in ret]

        ret[1] = ret[1].reshape(-1, 1)

        ret[-1] = ret[-1].reshape(-1, 1)
        ret = [th.from_numpy(r).to(self.device) for r in ret]
        ret.insert(1, th.from_numpy(np.asarray(actions)).to(self.device))
        ret.append(infos)
        return ret

    def sample(self, batch_size: int, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class GoalReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, device='cpu', vectorized=True):
        super(GoalReplayBuffer, self).__init__(size, device, vectorized)
        self.temp = []

    def add(self, obs_t, action, reward, obs_tp1, done, info):
        data = (obs_t, action, reward, obs_tp1, done, info)
        self.temp.append(data)

        if done:
            if self.vectorized:
                succ = info[0]["is_success"]
                for history in self.temp:
                    for o_t, a, r, o_tp1, d, info in zip(*history):
                        self._add(o_t, a, r, o_tp1, d, float(succ))
            else:
                succ = info["is_success"]
                for history in self.temp:
                    o_t, a, r, o_tp1, d, info = history
                    self._add(o_t, a, r, o_tp1, d, float(succ))
            self.temp = []

    def _encode_sample(self, idxes: Union[List[int], np.ndarray], ):
        obses_t, actions, rewards, obses_tp1, dones, infos = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, info = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            infos.append(info)
        ret = [obses_t, rewards, obses_tp1, dones, infos]
        ret = [np.asarray(r, dtype=np.float32) for r in ret]

        ret[1] = ret[1].reshape(-1, 1)
        ret[-2] = ret[-2].reshape(-1, 1)
        ret[-1] = ret[-1].reshape(-1, 1)
        ret = [th.from_numpy(r).to(self.device) for r in ret]
        ret.insert(1, th.from_numpy(np.asarray(actions)).to(self.device))

        return ret
