# Utilities

This folder contains code to evaluate the whole pipeline and to reshuffle the nvidia pure datasets.

## Usage
For both scripts the working directory should be `utils`

Running `python metrics_calculation.py` will calculate the evaluation metrics of all models on their respective test-set. 
The results will be printed into the commandline.

Running `python create_new_dataset_configuration.py` will shuffle the nvidia pure dataset trajectories 
and randomly split it into train/validation + test set.