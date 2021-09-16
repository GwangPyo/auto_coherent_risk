import pandas as pd
import numpy as np


def read():
    df = pd.read_csv("markov_chain_test.csv")
    obs = df["obs"].values
    done = df["done"].values
    next_obs = df["next_obs"].values
    episode_id = df["episode_id"].values
    time_step = df["time_step"].values
    # preprocess obs

    one_hot_obs = np.zeros(shape=(len(df), 7))
    one_hot_obs[np.arange(obs.shape[0]), obs] = 1
    one_hot_next_obs = np.zeros(shape=(len(df), 7))
    one_hot_next_obs[np.arange(obs.shape[0]), next_obs] = 1
    one_hot_next_obs[np.where(next_obs == -1), -1] = 0
    return one_hot_obs, done, one_hot_next_obs, episode_id, time_step


class MarkovChainData(object):
    def __init__(self):
        self.obs, self.done, self.next_obs, self.episode_id, self.time_step = read()
        self.n_episodes = self.episode_id[-1]
        x = [(np.where(self.episode_id == i)[0]) for i in range(self.n_episodes)]
        # [start, end)
        self.episode_index = np.asarray([(np.min(i), np.max(i) + 1) for i in x], dtype=np.int64)

    def __getitem__(self, index):
        return self.obs[index], self.done[index], self.episode_id[index], self.time_step[index]

    def get_random_episode(self, num_episode, reverse: bool):
        n_episode = np.random.randint(0, num_episode, size=(num_episode,))
        indices = []
        for idx in n_episode:
            start = self.episode_index[idx][0]
            end = self.episode_index[idx][1]

            arr_slice = np.arange(start, end)
            if reverse:
                arr_slice = np.flip(arr_slice)
            indices.append(arr_slice)
        indices = np.concatenate(indices)
        return self[indices]

    def get_partial_random(self, num_episode, reverse):
        n_episode = np.random.randint(0, num_episode, size=(num_episode, ))
        indices = []
        for idx in n_episode:
            start = self.episode_index[idx][0]
            end = np.random.randint(start + 2, self.episode_index[idx][1] + 1)
            arr_slice = np.arange(start, end)

            if reverse:
                arr_slice = np.flip(arr_slice)
            indices.append(arr_slice)
        indices = np.concatenate(indices)
        return self[indices]
