import h5py
import lightning
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


class BioTacDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.data[idx],
                                                                             dtype=torch.float)  # autoencoder -> input == ground_truth


class BioTacDataPreperation(lightning.LightningDataModule):
    def __init__(self, data_paths, test_data_path, train=0.80, batch_size=64, store_norm=False, norm_path="./norm.pt", indenter_filter=None,
                 shuffel_seed=0, experiment_postfix="all_data"):
        super().__init__()
        self.norm_path = norm_path
        self.std = None
        self.mean = None
        self.val_data = None
        self.test_data = None
        self.train_data = None
        self.data = None
        self.pre_test_data = None
        self.data_paths = data_paths
        self.test_data_path = test_data_path
        self.train_precentage = train
        self.batch_size = batch_size
        self.store_norm = store_norm
        self.indenter_filter = indenter_filter
        self.experiment_postfix = experiment_postfix

        np.random.seed(shuffel_seed)

    def prepare_data(self):
        datasets = []
        indenter_types = []
        others = 9
        for data_path in self.data_paths:
            with h5py.File(data_path, "r") as f:
                datasets.append(f["electrode_vals"]["data"][:, :])
                if "indenter_type" in f.keys():
                    indenter_types.append(f["indenter_type"]["data"][:])
                else:
                    indenter_types.append(np.ones((f["electrode_vals"]["data"][:, :].shape[0], 1)) * others)
                    others += 1

        self.data = np.concatenate(datasets)
        indenter_types = np.concatenate(indenter_types)

        with h5py.File(self.test_data_path, "r") as f:
            self.pre_test_data = f["electrode_vals"]["data"][:, :]
            test_indenter_type = f["indenter_type"]["data"][:]

        if self.indenter_filter is not None:
            mask = np.argwhere(np.resize(indenter_types == self.indenter_filter, indenter_types.shape[0]))
            test_mask = np.argwhere(np.resize(test_indenter_type == self.indenter_filter, test_indenter_type.shape[0]))

            self.data = self.data[np.resize(mask, (mask.shape[0]))]
            self.pre_test_data = self.pre_test_data[np.resize(test_mask, (test_mask.shape[0]))]

    def setup(self, stage=None):
        np.random.seed(0)
        np.random.shuffle(self.data)

        train_split = int(self.data.shape[0] * self.train_precentage)
        val_split = self.data.shape[0] - train_split
        test_split = int(self.data.shape[0] * 0.10)

        self.std = np.std(self.data[:train_split], axis=0)
        self.mean = np.mean(self.data[:train_split], axis=0)

        if self.store_norm:
            norm_dict = {
                "std": torch.tensor(self.std),
                "mean": torch.tensor(self.mean)
            }
            torch.save(norm_dict, self.norm_path+f"value_norms_{self.experiment_postfix}.pt")

        self.train_data, self.test_data, self.val_data = (
            BioTacDataset((self.data[:train_split] - self.mean) / self.std),
            BioTacDataset((np.concatenate([self.pre_test_data, self.data[train_split:train_split+test_split]]) - self.mean) / self.std),
            BioTacDataset((self.data[train_split+test_split:] - self.mean) / self.std)
        )

        print(f"Train set size: {len(self.train_data)}")
        print(f"Test set size: {len(self.test_data)}")
        print(f"Val set size: {len(self.val_data)}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
