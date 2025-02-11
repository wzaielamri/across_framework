from pathlib import Path
import hydra
import lightning
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from across.BioTac_Signal_to_Deformation.src.dataset_loader.biotac_latent_space_loader import BioTacLatentSpaceModule
from across.BioTac_Signal_to_Deformation.src.model.projection_network import ProjectionNetworkBiotac
from across.Mesh_Reconstruction.src.model.fem_autoencoder import fem_VAE

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train_latent_space_translation(cfg):

    epochs = cfg["experiment"]["epochs"]

    hparams = OmegaConf.create(cfg["model"])
    experiment_name = f"lr_{cfg['model']['lr']}_seed_{cfg['experiment']['seed']}"

    reconstruction_model = fem_VAE.load_from_checkpoint(
        cfg['dataset']['deformation_checkpoint'],
        mean=0, std=1, z=128)
    reconstruction_model = reconstruction_model.to("cuda:0")

    use_mu = cfg['experiment']['use_mu']

    data_module = BioTacLatentSpaceModule(data_file=cfg['dataset']['dataset_file'], batch_size=int(cfg['model']['batch_size']), num_workers=cfg['experiment']['num_workers'], use_mu=use_mu)

    norm = torch.load(cfg['dataset']['deformation_norm'])

    if cfg["experiment"]["decoder_evaluation"]:
        model = ProjectionNetworkBiotac(hparams, reconstruction_model, norm)
    else:
        model = ProjectionNetworkBiotac(hparams)

    model = model.to(cfg["experiment"]["device"])
    project_name = cfg["experiment"]["name"]+"_"+cfg["dataset"]["data_postfix"]
    logger = WandbLogger(name=experiment_name,
                         project=project_name, log_model=True,
                         save_dir=str(Path("logs").resolve()))

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=50, verbose=True, )
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg["experiment"]["checkpoint_path"]).resolve() / cfg["dataset"]["data_postfix"] / experiment_name,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    trainer = lightning.Trainer(max_epochs=int(epochs), logger=logger, default_root_dir=f"logs/lightning", callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model=model, datamodule=data_module)

    metrics = trainer.validate(model=model, datamodule=data_module)
    print(metrics)
    metrics = trainer.test(model=model, datamodule=data_module)
    print(metrics)
    wandb.finish()
    return metrics[0]


if __name__ == '__main__':
    train_latent_space_translation()
