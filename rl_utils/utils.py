import torch as th
from typing import Union
import gym
import numpy as np
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import os


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


def _evaluate(env, model, steps=1000, verbose=True):
    scores = []
    success = []
    iterator = tqdm(range(steps)) if verbose else range(steps)
    for _ in iterator:
        obs = env.reset()
        done = False
        score = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            score += reward
            if done and "is_success" in info.keys():
                success.append(info["is_success"])
        scores.append(score)
    if len(success) > 0:
        result = {"scores": scores, "success": success}
    else:
        result = {"scores": scores}
    return result


def result_to_csv(path, result_dict):
    df = pd.DataFrame.from_dict(result_dict)
    df.to_csv(path)
    return df


def evaluate(env, model, save_path, steps=1000, verbose=True):
    result = _evaluate(env, model, steps, verbose)
    df = result_to_csv(save_path, result)
    print(df)


def read_csv(path, keys):
    df = pd.read_csv(path)
    if isinstance(keys, str):
        frame = df[keys]
        return np.mean(frame)
    try:
        rets = {}
        for k in keys:
            rets[k] = read_csv(path, k)
        return rets
    except TypeError:
        exit(-1)


def read_csv_in_folders(folder, keys, suffix=".csv"):
    filenames = os.listdir(folder)
    results = []
    for name in filenames:
        try:
            match = (name[-len(suffix):] == suffix)
            if match:
                results.append((name, read_csv(path=f"{folder}/{name}", keys=keys)))
        except IndexError:
            pass
    results.sort(key=lambda x: x[0])
    return results


