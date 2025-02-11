# BioTac_Signal_Reconstruction

This package contains scripts to train a VAE for the purpose of Reconstructing the 19 BioTac Signals

## Usage 

The model can be trained with `python train_biotac.py` when the working directory is `across/BioTac_Signal_Reconstruction/src`

The training can be configured in `across/BioTac_Signal_Reconstruction/configs/main.yaml`. There is one preset for the dataset:

- `all`: nvidia + ruppel data

This model can be used in combination with the BioTac Mesh VAE to train a Translation Network.

The performance and results of the resulting model can be explored in `across/BioTac_Signal_Reconstruction/latent_space_traversal.ipynb`