from pathlib import Path

import hydra
from lightning.pytorch.loggers import WandbLogger
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from across.BioTac_to_DIGIT_Deformation.src.dataset_loader.latent_space_data_loader import LatentSpaceDataset
from across.BioTac_to_DIGIT_Deformation.src.model.projection_network import \
    ProjectionNetworkDeformation
from across.BioTac_to_DIGIT_Deformation.src.utils.get_best_smac_config import get_best_smac_config
from across.Mesh_Reconstruction.src.model.fem_autoencoder import fem_VAE

import random
import numpy as np

import os
from omegaconf import OmegaConf

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torch_geometric.loader import DataLoader

os.environ["WANDB_INIT_TIMEOUT"] = "300"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed != -1:
        # the seed is the jobid which has . in it
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
    # len(args.devices.split(","))

    num_workers = num_workers * num_devices
    devices = cfg["experiment"]["devices"]


    print("Available GPUs: ", num_devices)
    print("Devices: ", devices)
    experiment_seed=cfg["experiment"]["seed"]
    g = set_seed(experiment_seed)

    seed_biotac = cfg["dataset"]["seed_biotac"]
    seed_digit = cfg["dataset"]["seed_digit"]
    latent_space_path=cfg["dataset"]["latent_space_path"]
    use_mu = cfg["dataset"]["use_mu"]

    latent_space_datapaths_biotac = [f"{latent_space_path}biotac_{seed_biotac}_mesh_z_train_L_1_{cfg['dataset']['file_postfix']}_mu.npy",
                                     f"{latent_space_path}biotac_{seed_biotac}_mesh_z_val_L_1_{cfg['dataset']['file_postfix']}_mu.npy"]
    latent_space_datapaths_digit = [f"{latent_space_path}digit_{seed_digit}_mesh_z_train_L_1_{cfg['dataset']['file_postfix']}_mu.npy",
                                    f"{latent_space_path}digit_{seed_digit}_mesh_z_val_L_1_{cfg['dataset']['file_postfix']}_mu.npy"]

    # this is for mse loss validation on the fem mesh
    fem_vae_checkpoint = cfg["dataset"]["fem_checkpoint"]

    # allData
    dataset_train = LatentSpaceDataset(latent_space_datapaths_biotac[0], latent_space_datapaths_digit[0], use_mu=use_mu)
    dataset_val = LatentSpaceDataset(latent_space_datapaths_biotac[1], latent_space_datapaths_digit[1], use_mu=use_mu)

    print("Dataset: Train", len(dataset_train), "- Val", len(dataset_val))

    # get the dict in cfg["model"] to conf
    conf = OmegaConf.to_container(cfg["model"], resolve=True)
    print("Config: ", conf)
    conf["note"] = cfg['dataset']['file_postfix'] + "_" + str(experiment_seed)

    conf["device"] = devices
    conf["seed"] = str(seed_biotac) + "_" + str(seed_digit)
    conf["z_input"] = len(dataset_train[0][0])
    conf["z_output"] = len(dataset_train[0][1])
    conf["fem_vae_checkpoint"] = fem_vae_checkpoint

    if experiment_seed != -1:
        train_loader = DataLoader(dataset_train, batch_size=conf["batch_size"], shuffle=True, num_workers=num_workers,
                                  worker_init_fn=seed_worker, generator=g, )
        val_loader = DataLoader(dataset_val, batch_size=conf["batch_size"], shuffle=False, num_workers=num_workers,
                                worker_init_fn=seed_worker, generator=g, )
    else:
        train_loader = DataLoader(dataset_train, batch_size=conf["batch_size"], shuffle=True, num_workers=num_workers, )
        val_loader = DataLoader(dataset_val, batch_size=conf["batch_size"], shuffle=False, num_workers=num_workers, )

    print("Created Dataloaders")


    # for the interpretebale loss on the mesh output:
    fem_vae_model = fem_VAE.load_from_checkpoint(fem_vae_checkpoint, mean=0, std=1)  # mean and std are not needed for encoding
    fem_vae_model.eval()
    fem_vae_model = fem_vae_model.to(devices[0]) # cuda:0
    print("Loaded FEM VAE Model")

    model = ProjectionNetworkDeformation(hparams=conf, reconstruction_model=fem_vae_model)  # validate only on the targetMesh

    norm = torch.load(cfg["dataset"]["norm_path"])

    model.update_mean_std(mean=norm["mean"], std=norm["std"])

    print("Created Model")
    print(model)

    experiment_name = f"batch_size_{conf.get('batch_size')}:lr_{conf.get('lr')}_seed_{conf.get('seed')}_{conf.get('note')}"

    logger = WandbLogger(name=experiment_name, project="fem_projection",
                         log_model=True,
                         save_dir=str(Path("logs").resolve()))

    # save checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg["experiment"]["checkpoint_storage"] + experiment_name + "/",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=50, min_delta=0.001, verbose=True, )

    trainer = pl.Trainer(accelerator="gpu", max_epochs=cfg["experiment"]["epochs"], logger=logger, default_root_dir=f"logs/lightning", callbacks=[checkpoint_callback, lr_monitor,
                                                                                       early_stop_callback], )  # deterministic=True if seed != -1 else False) #TODO: set deterministic to True gives cuda memory error

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    main()
