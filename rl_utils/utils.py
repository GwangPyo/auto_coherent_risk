import torch as th
from typing import Union
import gym
import numpy as np
from typing import List, Dict
from datetime import timedelta
import os
from torch.utils.tensorboard import SummaryWriter


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


def information_handler(info_dict_list: List[Dict]):
    returned_list = []
    time_deltas = []
    fps = []
    length = []
    rewards = []
    for dictionary in info_dict_list:
        # check item sizes
        if len(dictionary.items()) == 0:
            continue
        else:
            if "terminal_observation" in dictionary.keys():
                dictionary.pop("terminal_observation")
            if "episode" in dictionary.keys():
                v = dictionary["episode"]
                length.append(v.pop("l"))
                rewards.append(v.pop("r"))
                t = v.pop("t")
                time_deltas.append(timedelta(seconds=t))
                returned_list.append(dictionary)
    time_deltas = np.mean(time_deltas)
    time_info = {"episode/iter_time": time_deltas, "time/fps": fps}
    return returned_list,  time_info


class DummyWriter(object):
    def __init__(self, *args, **kwargs):
        pass

    def logkv(self, *args, **kwargs):
        pass

    def logkvs(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def make_writer(root_directory_name, tb_log_name, writing_option='Force'):
    if writing_option == 'Force' or 'ask':
        try:
            os.listdir(root_directory_name)
        except FileNotFoundError:
            if writing_option == 'force':
                os.mkdir(root_directory_name)
            elif writing_option == 'ask':
                while True:
                    print(f"There is no directory named {root_directory_name}")
                    option = input("do you want to make the directory?[Yes/No]")
                    if option == 'Yes' or option == 'yes' or option == 'y' or option == 'Y':
                        os.mkdir(root_directory_name)
                        break
                    elif option == 'No' or option == 'no' or option == 'n' or option == 'N':
                        print("Do not make writer")
                        make_writer(root_directory_name, tb_log_name, writing_option="dummy")
                        break
                    else:
                        continue
        i = 0
        while True:
            try:
                os.mkdir(f"{root_directory_name}/{tb_log_name}_{i}")
            except FileExistsError:
                i += 1
                continue
            break
        return SummaryWriter(log_dir=f"{root_directory_name}/{tb_log_name}_{i}")
    else:
        return DummyWriter()
