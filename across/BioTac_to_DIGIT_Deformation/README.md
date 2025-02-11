# BioTac_to_DIGIT_Deformation

This package contains code to train an MLP that maps BioTac latent space to DIGIT latent space. 

## Usage 

The model can be trained using `python train_projection.py` while using `across/BioTac_to_DIGIT_Deformation/src` as the working directory.

By running `python transfer_fem.py` the BioTac latent space of the test set are decoded and converted to DIGIT/ BioTac deformations.

## Note

The training latent spaces should be already generated and saved in the `../../Data/datasets/latent_space` directory using the `Mesh_Reconstruction/src/encode_fem.py` script.