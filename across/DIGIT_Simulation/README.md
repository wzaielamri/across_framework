# DIGIT_Simulation

This package is adapted from Taxim-Robot: An extension of Taxim simulation model.
We adjust this package and use it to generate DIGIT output images based on DIGIT 3D mesh deformations.


## Usage

First run the script: ```python metrics_calculation.py``` this script should be run from it directory, to avoid any path issues: script is found under the utils directory in the root of the project.

The script will calculate the metrices and additionally save the DIGIT 3D mesh deformations needed for image generation.

Then run the script: ```python generate_transfer_data.py```

The script will generate the images based on the DIGIT 3D mesh deformations.



## License
This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.


