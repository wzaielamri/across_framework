from pathlib import Path

import hydra
import lightning
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from across.BioTac_Signal_Reconstruction.src.model.biotac_autoencoder import BioTacAutoencoder
from across.BioTac_Signal_Reconstruction.src.datasets.data_loader import BioTacDataPreperation

from omegaconf import OmegaConf
import wandb
import numpy as np


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def train_biotac(cfg):
    data_module = BioTacDataPreperation(
        data_paths=cfg["dataset"]["train_data"],
        test_data_path=cfg["dataset"]["test_data"],
        batch_size=cfg["model"]["batch_size"],
        indenter_filter=cfg["dataset"]["indenter_filter"],
        store_norm=cfg['experiment']['store_norm'],
        norm_path=cfg['experiment']['norm_path'],
        shuffel_seed=cfg['experiment']['seed'],
        experiment_postfix=cfg['dataset']['data_postfix']
    )

    data_module.prepare_data()
    data_module.setup("0")

    param_dict = cfg["model"]
    # param_dict["train_size"] = len(data_module.train_data)

    hparams = OmegaConf.create(param_dict)
    model = BioTacAutoencoder(hparams)
    logger = WandbLogger(name=f"kl_weight_{hparams.get('kl_weight')}:lr_{hparams.get('lr')}_ep_{cfg['experiment']['epochs']}",
                         project=f"{cfg['experiment']['project_name']}_{cfg['dataset']['data_postfix']}", log_model=True,
                         save_dir=str(Path("../logs").resolve()))

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", patience=10, verbose=True, )
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg["experiment"]["checkpoint_path"]).resolve() / cfg["dataset"]["data_postfix"],
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    trainer = lightning.Trainer(max_epochs=cfg['experiment']['epochs'], logger=logger, default_root_dir=f"logs/lightning", callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(model=model, datamodule=data_module)

    metrics = trainer.validate(model=model, datamodule=data_module)
    print(metrics)
    metrics = trainer.test(model=model, datamodule=data_module)
    print(metrics)
    wandb.finish()
    return metrics[0]


if __name__ == '__main__':
    np.random.seed(0)
    res = train_biotac()
