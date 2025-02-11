import hydra
import lightning as pl

import torch
from across.Mesh_Reconstruction.src.dataset_loader.mesh_data_loader import FEMDataset
from torch_geometric.loader import DataLoader
from across.Mesh_Reconstruction.src.dataset_loader.transform import Normalize

import random
import numpy as np

import argparse
import os
import copy

os.environ["WANDB_INIT_TIMEOUT"] = "300"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed != -1:
        # the seed is the jobid which has . in it
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed, workers=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        g = torch.Generator()
        g.manual_seed(0)
        return g
    else:
        print("Not setting seed.")
        return None

@hydra.main(version_base=None, config_path="../configs", config_name="encode")
def main(cfg):
    # get number of cores available
    available_cores = os.cpu_count()


    checkpoint_path = cfg["checkpoint"]["path"]
    print(checkpoint_path)
    sensor_name = cfg["checkpoint"]["sensor_name"]

    output_path = cfg["run"]["output_path"]
    # extract the seed from the checkpoint path
    seed = int(checkpoint_path.split("seed_")[1].split("_")[0])
    L = cfg["run"]["L"]

    num_workers = cfg["run"]["num_workers"] if cfg["run"]["num_workers"] != -1 else available_cores

    num_devices = torch.cuda.device_count()
    # len(args.devices.split(","))

    num_workers = num_workers * num_devices
    network_name = cfg["checkpoint"]["network_name"]

    print("Available GPUs: ", num_devices)
    print("Devices: ", cfg["run"]["devices"])
    print("Sensor: ", cfg["checkpoint"]["sensor_name"])
    print("Seed: ", seed)
    note = cfg["dataset"]["name_postfix"]

    print("Data will be saved under: ", f"{output_path}{sensor_name}_{seed}_mesh_z_train_L_{L}{note}_mu.npy")

    g = set_seed(seed)

    normalize_transform = Normalize()
    if int(cfg["dataset"]["envs_to_use"]) == 0:
        dataset_train = FEMDataset(cfg["dataset"]["dataset_path"], dtype='train', pre_transform=normalize_transform, mesh_path=cfg["checkpoint"]["mesh_path"],
                                sensor_names=[cfg["checkpoint"]["sensor_name"]], dataset_len=cfg["dataset"]["dataset_length"], envs_to_use=[cfg["dataset"]["envs_to_use"]])
        dataset_val = FEMDataset(cfg["dataset"]["dataset_path"], dtype='val', pre_transform=normalize_transform, mesh_path=cfg["checkpoint"]["mesh_path"],
                                sensor_names=[cfg["checkpoint"]["sensor_name"]], dataset_len=cfg["dataset"]["dataset_length"], envs_to_use=[cfg["dataset"]["envs_to_use"]])
        dataset_test = FEMDataset(cfg["dataset"]["dataset_path"], dtype='test', pre_transform=normalize_transform, mesh_path=cfg["checkpoint"]["mesh_path"],
                                sensor_names=[cfg["checkpoint"]["sensor_name"]], dataset_len=cfg["dataset"]["dataset_length"], envs_to_use=[cfg["dataset"]["envs_to_use"]])
    else:   
        dataset_train = FEMDataset(cfg["dataset"]["dataset_path"], dtype='train', pre_transform=normalize_transform, mesh_path=cfg["checkpoint"]["mesh_path"],
                                sensor_names=[cfg["checkpoint"]["sensor_name"]], dataset_len=cfg["dataset"]["dataset_length"])
        dataset_val = FEMDataset(cfg["dataset"]["dataset_path"], dtype='val', pre_transform=normalize_transform, mesh_path=cfg["checkpoint"]["mesh_path"],
                                sensor_names=[cfg["checkpoint"]["sensor_name"]], dataset_len=cfg["dataset"]["dataset_length"])
        dataset_test = FEMDataset(cfg["dataset"]["dataset_path"], dtype='test', pre_transform=normalize_transform, mesh_path=cfg["checkpoint"]["mesh_path"],
                                sensor_names=[cfg["checkpoint"]["sensor_name"]], dataset_len=cfg["dataset"]["dataset_length"])


    print("Dataset: Train", len(dataset_train), "- Val", len(dataset_val), "- Test", len(dataset_test))

    # no shuffle needed we only encode the data and bs is 1
    if seed != -1:
        train_loader = DataLoader(dataset_train, batch_size=cfg["run"]["batch_size"], shuffle=False, num_workers=num_workers,
                                  worker_init_fn=seed_worker, generator=g, )
        val_loader = DataLoader(dataset_val, batch_size=cfg["run"]["batch_size"], shuffle=False, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=g, )
        test_loader = DataLoader(dataset_test, batch_size=cfg["run"]["batch_size"], shuffle=False, num_workers=num_workers,
                                 worker_init_fn=seed_worker, generator=g, )
    else:
        train_loader = DataLoader(dataset_train, batch_size=cfg["run"]["batch_size"], shuffle=False, num_workers=num_workers, )
        val_loader = DataLoader(dataset_val, batch_size=cfg["run"]["batch_size"], shuffle=False, num_workers=num_workers, )
        test_loader = DataLoader(dataset_test, batch_size=cfg["run"]["batch_size"], shuffle=False, num_workers=num_workers, )

    print("Created Config")
    # model = Nvidia_Autoencoder(conf,mean=dataset_train.mean,std=dataset_train.std)
    model_obj = getattr(__import__('across.Mesh_Reconstruction.src.model.fem_autoencoder', fromlist=[network_name]), network_name)
    # model = model_obj(conf,mean=dataset_train.mean,std=dataset_train.std)

    # print("Created Model")
    # print(model)

    # load the checkpoint
    model = model_obj.load_from_checkpoint(checkpoint_path, mean=0, std=1)  # mean and std are not needed for encoding
    print("Loaded Checkpoint from:", checkpoint_path)
    # encode the data
    model.eval()
    trainer = pl.Trainer(devices=num_devices, strategy="ddp", logger=None, default_root_dir=f"logs/lightning")
    # predict the data and save it
    # update the mean and std of the model for visualization
    model.update_mean_std(mean=dataset_train.mean, std=dataset_train.std)

    z_train = []
    z_val = []
    z_test = []

    for l in range(L):
        print("l:", l)

        output_train = trainer.predict(model, train_loader)
        for i in output_train:
            z_train.append([copy.deepcopy(i[1][0][0]), copy.deepcopy(i[2][0])])

        output_val = trainer.predict(model, val_loader)
        for i in output_val:
            z_val.append([copy.deepcopy(i[1][0][0]), copy.deepcopy(i[2][0])])

        output_test = trainer.predict(model, test_loader)
        for i in output_test:
            z_test.append([copy.deepcopy(i[1][0][0]), copy.deepcopy(i[2][0])])
    
    z_train = np.array(z_train)
    z_val = np.array(z_val)
    z_test = np.array(z_test)

    # save the output

    np.save(f"{output_path}{sensor_name}_{seed}_mesh_z_train_L_{L}{note}_mu.npy", z_train)
    np.save(f"{output_path}{sensor_name}_{seed}_mesh_z_val_L_{L}{note}_mu.npy", z_val)
    np.save(f"{output_path}{sensor_name}_{seed}_mesh_z_test_L_{L}{note}_mu.npy", z_test)


if __name__ == '__main__':
    main()
