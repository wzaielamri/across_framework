# DIGIT BioTac Isaac Gym

This package contains code to train a Translation Network between the Signal VAE latent space
and the BioTac Mesh VAE latent space

## Prerequisites

To train this Translation network it is necessary to first re-simulate the Nvidia_pure dataset.

## Usage 

1. Collect latent spaces of the dataset with `python convert_latent_spaces.py`
2. After configuring the config in `across/BioTac_Signal_to_Deformation/configs` the network can be trained by executing
`python train_latent_space_translation.py` if the working directory is `across/BioTac_Signal_to_Deformation/src`

After successfully training a network it's results can be explored with the Jupyter Notebook 
using `across/BioTac_Signal_to_Deformation/visualize_results.ipynb` 