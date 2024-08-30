# Stick Insect Proprioception

This repository contains the code for the research papers:
- "Encoding of movement primitives and body posture through distributed proprioception in walking and climbing insects," submitted to PLOS Computational Biology.
- "A spiking neural network model for proprioception of limb kinematics in insect locomotion," submitted to PLOS Computational Biology.

Running the code reproduces all the results presented in the articles.

## Requirements

At least 16GB of RAM, 32GB preferred. 

## Installation

Follow these steps to install and set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/ThomasvanderVeen/StickInsectProprioception

2. Navigate the project directory
    ```bash
   cd your-repository

3. Install dependencies
   ```bash
   pip install -r requirements.txt

## Usage

### General.py

This file allows users to modify several parameters:

- **N_SIMULATIONS**: Set the number of simulations to run (maximum 78, default is 78).
- **N_HAIRS**: Set the number of hairs in a hair field (default is 50).
- **pad**: Set the padding of figures in mm (default is 3).
- **save_data**: Choose whether to save data for subsequent use or use pre-saved data (default is True).
- **run_optimization**: Choose whether to run optimization steps or use pre-saved data. Set to False for faster performance after the first run (default is True).

- **custom_colors, custom_markers, custom_linestyles**: Customize these for the figures.
- **plot_parameters**: Adjust the plotting parameters for the figures.

Note: Constants cannot be changed and keep the default settings to yield the same results as in the research articles

### 0_run_all_notebooks.ipynb

This notebook runs all other notebooks in sequence. Options for saving data and running optimizations can be set individually for each notebook.

### Notebooks

Run the following notebooks in order as outlined in `0_run_all_notebooks.ipynb`:

1. `prepare_data.py`: Prepares the data; should be run once after installation.
2. `1_afferents.ipynb`
3. `2_motion_neurons.ipynb`
4. `2_position_neurons.ipynb`
5. `3_motion_primitive_neurons.ipynb`
6. `4_posture_neurons.ipynb`


## License 

This project is licensed under the MIT License.


  

