import numpy as np
import torch
from torch.utils.data import Dataset


class LatentSpaceDataset(Dataset):
    def __init__(self, data_dir_biotac, data_dir_digit, input_sensor="biotac", use_mu=False):
        super(LatentSpaceDataset, self).__init__()
        self.input_sensor = input_sensor
        self.data_dir_biotac = data_dir_biotac
        self.data_biotac = np.load(self.data_dir_biotac)
        self.data_biotac = self.data_biotac.reshape(-1, self.data_biotac.shape[-1])
        self.data_biotac = torch.tensor(self.data_biotac, dtype=torch.float32)
        self.data_dir_digit = data_dir_digit
        self.data_digit = np.load(self.data_dir_digit)
        self.data_digit = self.data_digit.reshape(-1, self.data_digit.shape[-1])
        self.data_digit = torch.tensor(self.data_digit, dtype=torch.float32)

    def __len__(self):
        return self.data_biotac.shape[0]

    def __getitem__(self, idx):
        if self.input_sensor == "biotac":
            return self.data_biotac[idx], self.data_digit[idx]
        if self.input_sensor == "digit":
            return self.data_digit[idx], self.data_biotac[idx]
