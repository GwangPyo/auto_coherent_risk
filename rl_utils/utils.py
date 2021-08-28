import torch as th
from typing import Union
import gym
import numpy as np


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.
    :param device: One for 'auto', 'cuda', 'cpu'
    :return:
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


def dump_state_dict(source, target):
    state_dict = source.state_dict()
    target.load_state_dict(state_dict)


def polyak_update(source, target, tau):
    """
    Cherry-picked from
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py
    and slightly modified
    """
    source_parameters = source.parameters()
    target_parameters = target.parameters()
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(source_parameters, target_parameters):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def dimension(space: gym.spaces.Box):
    return np.prod(space.shape)


class DummyEnv(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
