import argparse
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
import h5py
import os
from psbody.mesh import Mesh

from across.Mesh_Reconstruction.src.model.operations.mesh_operations import get_vert_connectivity
from across.Mesh_Reconstruction.src.dataset_loader.transform import Normalize
from across.Mesh_Reconstruction.src.utils.convert_tet_to_msh import conv_tet_to_vertices_and_faces


class FEMDataset(InMemoryDataset):
    def __init__(self, root_dir, dtype='train', transform=None, pre_transform=None, mesh_path=None, sensor_names=[], dataset_len=-1, min_distance=0.1, max_distance=2.0, envs_to_use=range(0, 9)):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_transform = pre_transform
        self.sensor_names = sensor_names
        assert "digit" in self.sensor_names or "biotac" in self.sensor_names, "Sensor names should be either digit or biotac"

        self.dataset_len = dataset_len

        self.min_distance = min_distance * 1e-03
        self.max_distance = max_distance * 1e-03
        self.envs_to_use = np.array(envs_to_use)

        # get template_faces
        vertices, faces = conv_tet_to_vertices_and_faces(mesh_path)
        template_mesh = Mesh(v=vertices, f=faces)
        self.template_mesh_faces = template_mesh.f
        self.template_mesh_vertices = template_mesh.v

        self.dataset_pairs_path = os.path.join(root_dir, "biotac_digit_deformations/results/filtered_dataset_pairs.npy") 
        if self.dataset_len == -1:
            self.data_pairs = np.load(self.dataset_pairs_path)
            self.dataset_len = len(self.data_pairs)
        else:
            self.data_pairs = np.load(self.dataset_pairs_path)[:self.dataset_len]

        # random shuffle the data pairs
        # the data are aleardy shuffled when extracting with seed 0 and numpy.random.shuffle Do not shuffle to have the same test set for biotac and digit
        #np.random.shuffle(self.data_pairs)
        
        # filter the data pairs based on the distance

        self.data_file =[]
        for i in tqdm(range(len(self.data_pairs))):
            job_ind, job_name, trial_ind, env_ind, distance, digit_step_ind, biotac_step_ind = self.data_pairs[i]
            for sensor_name in self.sensor_names:
                self.data_file.append(f"{self.root_dir}/biotac_digit_deformations/results/results_{sensor_name}_{job_name}.hdf5")

        super(FEMDataset, self).__init__(root_dir, transform, pre_transform)
        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'val':
            data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")
        
        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']
        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]
        

    def read_element(self, data_pair):
        job_ind, job_name, trial_ind, env_ind, distance, digit_step_ind, biotac_step_ind = data_pair
        for sensor_name in self.sensor_names:
            with h5py.File(f"{self.root_dir}/biotac_digit_deformations/results/results_{sensor_name}_{job_name}.hdf5", "r") as container:
                if sensor_name == "biotac":
                    nodal_point = container[f"trial_{trial_ind}"][f"step_{biotac_step_ind}"][f"nodal_coords_{sensor_name}"][f"env_{env_ind}"][:]
                    force_vec = container[f"trial_{trial_ind}"][f"step_{biotac_step_ind}"][f"net_force_vecs_{sensor_name}"][f"env_{env_ind}"][:]
                elif sensor_name == "digit":
                    nodal_point = container[f"trial_{trial_ind}"][f"step_{digit_step_ind}"][f"nodal_coords_{sensor_name}"][f"env_{env_ind}"][:]
                    force_vec = container[f"trial_{trial_ind}"][f"step_{digit_step_ind}"][f"net_force_vecs_{sensor_name}"][f"env_{env_ind}"][:]
                else:
                    raise ValueError("Sensor name not recognized")
        return torch.tensor(nodal_point, dtype=torch.float), torch.tensor(force_vec, dtype=torch.float), torch.tensor(float(distance), dtype=torch.float), torch.tensor(int(env_ind), dtype=torch.int) 
    
    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt', 'distances.pt', 'envs.pt']
        env_string = '_'.join([str(e) for e in self.envs_to_use])
        processed_files = [f'{self.sensor_names[0]}_{self.dataset_len}_dist_{int(self.min_distance*10000)}_{int(self.max_distance*10000)}_env_{env_string}_{pf}' for pf in processed_files]
        #print(processed_files)
        return processed_files

    def save_obj_file(self, vertices, faces, filename):
        with open(filename, 'w') as f:
            for v in vertices:
                f.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for p in faces:
                f.write('f %d %d %d\n' % (p[0]+1, p[1]+1, p[2]+1))

    def read_obj_file(self, filename):
        vertices = []
        faces = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('f '):
                    faces.append(list(map(int, line.strip().split()[1:])))
        return vertices, faces

    def process(self):
        train_data, val_data, test_data = [], [], []
        train_distances, val_distances, test_distances = [], [], []
        train_envs, val_envs, test_envs = [], [], []
        train_vertices = []
        idx = 0
        for data_pair in tqdm(self.data_pairs):
            mesh_verts, force_vec, distance, env = self.read_element(data_pair)
            # check if the distance is within the range
            if distance < self.min_distance or distance > self.max_distance or int(env) not in self.envs_to_use:
                continue

            idx += 1
            adjacency = get_vert_connectivity(mesh_verts, self.template_mesh_faces).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

            if idx % 100 <= 10:
                test_data.append(data)
                test_distances.append(distance)
                test_envs.append(env)
            elif idx % 100 <= 20:
                val_data.append(data)
                val_distances.append(distance)
                val_envs.append(env)
            else:
                train_data.append(data)
                train_vertices.append(mesh_verts)
                train_distances.append(distance)
                train_envs.append(env)

        mean_train = torch.Tensor(np.mean(train_vertices, axis=0))
        std_train = torch.Tensor(np.std(train_vertices, axis=0))
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_transform.mean is None:
                    self.pre_transform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_transform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])
        distances_dict = {'train': torch.tensor(train_distances), 'val': torch.tensor(val_distances), 'test': torch.tensor(test_distances)}
        envs_dict = {'train': torch.tensor(train_envs), 'val': torch.tensor(val_envs), 'test': torch.tensor(test_envs)}
        torch.save(distances_dict, self.processed_paths[4])
        torch.save(envs_dict, self.processed_paths[5])

def prepare_sliced_dataset(path, mesh_path):
    FEMDataset(path, pre_transform=Normalize(), mesh_path=mesh_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data preparation for Convolutional Mesh Autoencoders')
    parser.add_argument('-d', '--data_dir', default="../dataset", help='path where the downloaded data is stored')

    args = parser.parse_args()
    data_dir = args.data_dir
    prepare_sliced_dataset(data_dir, mesh_path="../meshes/biotac/int_ext_skin/int_ext_skin_combined_ftw.tet",sensor_names = ["biotac"])
