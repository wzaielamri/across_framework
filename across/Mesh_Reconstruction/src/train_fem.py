from pathlib import Path

import hydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from across.Mesh_Reconstruction.src.dataset_loader.mesh_data_loader import FEMDataset
from torch_geometric.loader import DataLoader
from across.Mesh_Reconstruction.src.dataset_loader.transform import Normalize

import random
import numpy as np

import argparse
import os

from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

os.environ["WANDB_INIT_TIMEOUT"] = "300"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed != -1:
        # the seed is the jobid which has . in it; for condor only
        #seed = seed.replace(".", "")
        #seed = int(seed)
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


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg):
    # get number of cores available
    available_cores = os.cpu_count()

    num_workers = cfg["experiment"]["num_workers"] if cfg["experiment"]["num_workers"] != -1 else available_cores
    num_devices = torch.cuda.device_count()
    num_workers = num_workers * num_devices

    print("Available GPUs: ", num_devices)
    print("Devices: ", cfg["experiment"]["devices"])

    g = set_seed(cfg["experiment"]["seed"])

    normalize_transform = Normalize()
    if int(cfg["dataset"]["envs_to_use"]) == 0:
        dataset_train = FEMDataset(
            cfg["dataset"]["dataset_path"],
            dtype='train',
            pre_transform=normalize_transform,
            mesh_path=cfg["sensor"]["mesh_path"],
            sensor_names=[cfg["sensor"]["sensor_name"]],
            dataset_len=cfg["dataset"]["dataset_length"],
            envs_to_use=[cfg["dataset"]["envs_to_use"]])
        dataset_val = FEMDataset(
            cfg["dataset"]["dataset_path"],
            dtype='val',
            pre_transform=normalize_transform,
            mesh_path=cfg["sensor"]["mesh_path"],
            sensor_names=[cfg["sensor"]["sensor_name"]],
            dataset_len=cfg["dataset"]["dataset_length"],
            envs_to_use=[cfg["dataset"]["envs_to_use"]])
    else:
        dataset_train = FEMDataset(
            cfg["dataset"]["dataset_path"],
            dtype='train',
            pre_transform=normalize_transform,
            mesh_path=cfg["sensor"]["mesh_path"],
            sensor_names=[cfg["sensor"]["sensor_name"]],
            dataset_len=cfg["dataset"]["dataset_length"],)
        dataset_val = FEMDataset(
            cfg["dataset"]["dataset_path"],
            dtype='val',
            pre_transform=normalize_transform,
            mesh_path=cfg["sensor"]["mesh_path"],
            sensor_names=[cfg["sensor"]["sensor_name"]],
            dataset_len=cfg["dataset"]["dataset_length"],)
    
    print("Dataset: Train", len(dataset_train), "- Val", len(dataset_val))

    if cfg["experiment"]["seed"] != '-1':
        train_loader = DataLoader(
            dataset_train,
            batch_size=cfg["model"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=cfg["model"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        train_loader = DataLoader(
            dataset_train,
            batch_size=cfg["model"]["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=cfg["model"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
        )

    conf = dict(cfg["model"])
    conf["note"] = cfg["model"]["network"] + cfg["dataset"]["name_postfix"]
    conf["train_size"] = len(dataset_train)
    conf["val_size"] = len(dataset_val)
    conf["dataset_len"] = cfg["dataset"]["dataset_length"]
    conf["mesh_path"] = cfg["sensor"]["mesh_path"]
    conf["sensor_name"] = cfg["sensor"]["sensor_name"]
    conf["seed"] = cfg["experiment"]["seed"]
    conf["devices"] = cfg["experiment"]["devices"]
    conf = OmegaConf.create(conf)

    print("Created Config")
    # model = Nvidia_Autoencoder(conf,mean=dataset_train.mean,std=dataset_train.std)
    model_obj = getattr(__import__('across.Mesh_Reconstruction.src.model.fem_autoencoder', fromlist=[cfg["model"]["network"]]), cfg["model"]["network"])
    model = model_obj(conf, mean=dataset_train.mean, std=dataset_train.std)

    print("Created Model")
    print(model)
    experiment_name = f"{conf.get('sensor_name')}_batch_size_{conf.get('batch_size')}_kl_weight_{conf.get('kl_weight')}_cheb_order_{conf.get('cheby_order')}:z_{conf.get('z')}_lr_{conf.get('lr')}_L_{conf.get('L')}_seed_{conf.get('seed')}_{conf.get('note')}"

    logger = WandbLogger(name=experiment_name, project="fem_autoencoder",
                         log_model=True,
                         save_dir=str(Path("logs").resolve()))

    # save checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd() + "../../Data/checkpoints/"+ cfg["sensor"]["ckp_dir"] + "/" + experiment_name + "/",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=50, min_delta=0.0001, verbose=True, )
    trainer = pl.Trainer(accelerator="gpu", num_nodes=1, devices=num_devices, strategy="ddp", max_epochs=cfg["experiment"]["epochs"],
                         logger=logger, default_root_dir=f"logs/lightning", callbacks=[checkpoint_callback, lr_monitor,
                                                   early_stop_callback], )  # deterministic=True if seed != -1 else False) #TODO: set deterministic to True gives cuda memory error

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
