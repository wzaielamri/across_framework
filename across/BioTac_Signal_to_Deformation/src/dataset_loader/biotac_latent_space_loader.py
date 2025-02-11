import lightning
from torch.utils.data import DataLoader, Dataset
import torch


class BioTacLatentSpaceDataset(Dataset):
    def __init__(self, data_dict, use_mu=False):
        if use_mu:
            self.value_data = data_dict['value_mu']
            self.deformation_data = data_dict['deformation_mu']
        else:
            self.value_data = data_dict['value']
            self.deformation_data = data_dict['deformation']

    def __len__(self):
        return self.value_data.shape[0]

    def __getitem__(self, idx):
        return (self.value_data[idx].clone().detach(),
                self.deformation_data[idx].clone().detach())


class BioTacLatentSpaceModule(lightning.LightningDataModule):
    def __init__(self, data_file, batch_size=64, shuffle=True, use_mu=False, num_workers=4):
        super().__init__()
        self.use_mu = use_mu
        self.shuffle = shuffle
        self.test_data = None
        self.validation_data = None
        self.train_data = None
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        dataset_dict = torch.load(self.data_file)
        self.train_data = dataset_dict['train']
        self.validation_data = dataset_dict['validation']
        self.validation_data["value_mu"] = self.validation_data["value_mu"][:len(self.validation_data["value_mu"])]
        self.validation_data["deformation_mu"] = self.validation_data["deformation_mu"][:len(self.validation_data["deformation_mu"])]
        self.validation_data["value"] = self.validation_data["value"][:len(self.validation_data["value"])]
        self.validation_data["deformation"] = self.validation_data["deformation"][:len(self.validation_data["deformation"])]
        self.test_data = dataset_dict['test']

    def setup(self, stage=None):
        self.train_data = BioTacLatentSpaceDataset(self.train_data, use_mu=self.use_mu)
        self.validation_data = BioTacLatentSpaceDataset(self.validation_data, use_mu=self.use_mu)
        self.test_data = BioTacLatentSpaceDataset(self.test_data, use_mu=self.use_mu)

        print(f"Train Set Size: {len(self.train_data)}")
        print(f"Validation Set Size: {len(self.validation_data)}")
        print(f"Test Set Size: {len(self.test_data)}")
        print(f"Combined Set Size: {len(self.train_data) + len(self.validation_data) + len(self.test_data)}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


if __name__ == '__main__':
    module = BioTacLatentSpaceModule("../../../Data/datasets/value_def_dataset/accepted_dataset.pt")
    module.prepare_data()
    module.setup()
    print(next(iter(module.train_dataloader())))
