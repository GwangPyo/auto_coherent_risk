from torch.utils.data import Dataset
import pickle
import numpy as np
import torch as th


class TrajectoryDataSet(Dataset):
    def __init__(self, path, device='cpu'):
        super(TrajectoryDataSet, self).__init__()
        self.path = path
        self.device = device
        with open(self.path, "rb") as f:
            self.data_dict = pickle.load(f)
        self.obs = self.data_dict["obs"]
        self.action = self.data_dict["action"]
        self.done = self.data_dict["done"]
        self.next_obs = self.data_dict["next_obs"]
        self.episode_id = self.data_dict["episode_id"]
        self.time_step = self.data_dict["time_step"]
        self.success_label = self.data_dict["is_success"]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, index):
        obs = self.obs[index]
        action = self.action[index]
        next_obs = self.next_obs[index]
        done = self.done[index]
        success = self.success_label[index]
        time_step = self.time_step[index]
        return obs, action, next_obs, done, success, time_step

    def sample(self, batch_size):
        start = 0
        end = len(self) - batch_size
        start_index = np.random.randint(start, end)
        return self.to_torch(self[start_index: start_index + batch_size])

    def to_torch(self, tuple_of_array):
        return tuple([th.from_numpy(data).float().to(self.device) for data in tuple_of_array])

