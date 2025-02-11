import lightning
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BioTacValueToDeformationDataset(Dataset):
    def __init__(self, value_data, deformation_data, contact_loc_data, indenter_type_data, mesh_distance):
        self.mesh_distance = torch.tensor(mesh_distance, dtype=torch.float)
        self.value_data = torch.tensor(value_data, dtype=torch.float)
        self.deformation_data = torch.tensor(deformation_data, dtype=torch.float)
        self.contact_loc_data = torch.tensor(contact_loc_data, dtype=torch.float)
        self.indenter_type_data = torch.tensor(indenter_type_data, dtype=torch.int)

    def __len__(self):
        return self.value_data.shape[0]

    def __getitem__(self, idx):
        return (self.value_data[idx],
                self.deformation_data[idx],
                self.contact_loc_data[idx],
                self.indenter_type_data[idx],
                self.mesh_distance[idx])


class BioTacValueDeformationModule(lightning.LightningDataModule):
    def __init__(self, data_dirs, test_data_dir, value_norm, deformation_norm, batch_size=64, train_split=0.80):
        super().__init__()

        self.test_mesh_distance = None
        self.test_indenter_type = None
        self.test_contact_loc = None
        self.mesh_distance = None
        self.indenter_type = None
        self.contact_loc = None
        self.batch_size = batch_size
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.deformation_data = None
        self.value_data = None

        self.test_deformation_data = None
        self.test_value_data = None

        self.data_dirs = data_dirs  # [[value_file, deformation_file]....]
        self.test_data_dir = test_data_dir  # [value_file, deformation_file]
        self.value_norm = value_norm  # [mean, std]
        self.deformation_norm = deformation_norm  # [mean, std]
        self.train_split = train_split

    def prepare_data(self):
        full_value_data = []
        full_deformation_data = []
        full_contact_data = []
        full_indenter_type_data = []
        full_mesh_distance = []
        for data_dir in self.data_dirs:  # [value_dir, deformation_dir]
            with h5py.File(data_dir[0], "r") as f:
                full_value_data.append(f["electrode_vals"]["data"][:])
                full_contact_data.append(f["contact_loc"]["data"][:])
                full_indenter_type_data.append(f["indenter_type"]["data"][:])

            with h5py.File(data_dir[1], "r") as f:
                full_deformation_data.append(f["nodal_res"][:])
                full_mesh_distance.append(f["mesh_max_dist"][:])

        self.value_data = np.concatenate(full_value_data)
        self.deformation_data = np.concatenate(full_deformation_data)
        self.contact_loc = np.concatenate(full_contact_data)
        self.indenter_type = np.concatenate(full_indenter_type_data)
        self.mesh_distance = np.concatenate(full_mesh_distance)

        with h5py.File(self.test_data_dir[0], "r") as f:
            self.test_value_data = f["electrode_vals"]["data"][:]
            self.test_contact_loc = f["contact_loc"]["data"][:]
            self.test_indenter_type = f["indenter_type"]["data"][:]

        with h5py.File(self.test_data_dir[1], "r") as f:
            self.test_deformation_data = f["nodal_res"][:]
            self.test_mesh_distance = f["mesh_max_dist"][:]

        self.value_data = (self.value_data - self.value_norm[0]) / self.value_norm[1]
        self.test_value_data = (self.test_value_data - self.value_norm[0]) / self.value_norm[1]

        self.deformation_data = (self.deformation_data - self.deformation_norm[0]) / self.deformation_norm[1]
        self.test_deformation_data = (self.test_deformation_data - self.deformation_norm[0]) / self.deformation_norm[1]

    def setup(self, stage=None):
        train_split_index = int(self.train_split * len(self.value_data))
        #todo split through trajects
        self.train_data = BioTacValueToDeformationDataset(self.value_data[:train_split_index],
                                                          self.deformation_data[:train_split_index],
                                                          self.contact_loc[:train_split_index],
                                                          self.indenter_type[:train_split_index],
                                                          self.mesh_distance[:train_split_index])
        self.validation_data = BioTacValueToDeformationDataset(self.value_data[train_split_index:],
                                                               self.deformation_data[train_split_index:],
                                                               self.contact_loc[train_split_index:],
                                                               self.indenter_type[train_split_index:],
                                                               self.mesh_distance[train_split_index:])
        self.test_data = BioTacValueToDeformationDataset(self.test_value_data,
                                                         self.test_deformation_data,
                                                         self.test_contact_loc,
                                                         self.test_indenter_type,
                                                         self.test_mesh_distance)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


if __name__ == '__main__':
    print("Start")
    deformation_norm_dict = torch.load("../../dataset/norms/deformation_norms.pt")
    deformation_norm_s = np.array([deformation_norm_dict['mean'].numpy(), deformation_norm_dict['std'].numpy()])

    module = BioTacValueDeformationModule(
        data_dirs=[
            [
                "../../../BioTac_Signal_Reconstruction/data/BioTac_Pure_Exp_Dataset_v1.0/biotac1.hdf5",
                "../../../BioTac_Signal_Reconstruction/data/BioTac_Pure_Exp_Dataset_v1.0/biotac1_simulated.hdf5"
            ],
            [
                "../../../BioTac_Signal_Reconstruction/data/BioTac_Pure_Exp_Dataset_v1.0/biotac3.hdf5",
                "../../../BioTac_Signal_Reconstruction/data/BioTac_Pure_Exp_Dataset_v1.0/biotac3_simulated.hdf5"
            ]
        ],
        test_data_dir=[
            "../../../BioTac_Signal_Reconstruction/data/BioTac_Pure_Exp_Dataset_v1.0/biotac2.hdf5",
            "../../../BioTac_Signal_Reconstruction/data/BioTac_Pure_Exp_Dataset_v1.0/biotac2_simulated.hdf5"
        ],
        value_norm=np.load("../../dataset/norms/all/value_norms.npy"),
        deformation_norm=deformation_norm_s,
    )
    print("Prepare Data")
    module.prepare_data()
    print("Setup")
    module.setup()

