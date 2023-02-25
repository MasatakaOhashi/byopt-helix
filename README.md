# byopt-helix
byopt-helix is a software for optimizing the parameters of helical reconstruction in cryo-EM using Bayesian optimization.

## Usage
To run byopt-helix, use the following command:
```
python byopt-helix.py setting.txt
```
where setting.txt is a configuration file that contains the parameters for the Bayesian optimization.



## Configuration file format
This configuration file specifies the parameters for byopt-helix.

### General Parameters
|Parameter	| Description |
| ----------- | ----------- |
|project_dir| the project directory path |
|inputfile| the path to the input star file (ex. particles.star) |
|referencefile| the path to the reference model file (ex. featureless_init_model.mrc) |
|folder_name| the name of the output folder |
|script_name| the name of the script to run (3dclass.sh) |
|relion_gpu| the GPUs to use (ex. "0,1") |
|relion_thread| the number of threads to use |
|relion_iter| the number of iterations to run |
|relion_K| the number of classes to classify the particles into |
|relion_angpix| the pixel size (in Angstroms) of the input and reference files |
|relion_particle_diameter| the diameter of the particles (in Angstroms) |
|relion_helical_outer_diameter| the outer diameter of the helix (in Angstroms) |
|relion_helical_inner_diameter| the inner diameter of the helix (in Angstroms) |
|relion_helical_z_percentage| the percentage of the z-height of the helix (0 to 1) |
|relion_helical_nr_asu| the number of asymmetric units in the helix |
|relion_sym| the symmetry of the particles (default is C1) |
|relion_tau2_fudge| the fudge factor for the tau2 parameter |
|relion_ini_high| the initial high pass filter value (in Angstroms) |

### Bayesian Optimization Parameters
|Parameter	| Description |
| ----------- | ----------- |
|model_type| the type of model to use (GP_MCMC) |
|acquisition_type| the type of acquisition to use (EI_MCMC) |
|search_type| the type of search to use (1) |
|bayesian_opt_normalize_Y| whether to normalize the Y values (True or False) |
|bayesian_opt_acquisition_weight| the weight of the acquisition function (default is 1.0) |
|bayesian_opt_acquisition_jitter| the jitter value for the acquisition function (default is 10.0) |
|bayesian_opt_ARD| whether to use Automatic Relevance Determination (True or False) |
|bayesian_opt_lengthscale| the lengthscale values for the ARD kernel (default is 1.0 1.0) |
|bayesian_opt_1d_search_lengthscale| the lengthscale value for the 1D search |
|bayesian_opt_1d_search_ARD| whether to use ARD for the 1D search (True or False) |
|bayesian_opt_1d_search_first_max_itr| the maximum number of iterations for the first 1D search |
|bayesian_opt_1d_search_second_max_itr| the maximum number of iterations for the second 1D search |
|bayesian_opt_2d_search_max_itr| the maximum number of iterations for the 2D search |
|bayesian_opt_eps| the convergence threshold (default is 1e-3) |
|bayesian_opt_dp_start| the starting value of the helical twist |
|bayesian_opt_dp_end| the ending value of the helical twist |
|bayesian_opt_dp_num| the number of steps for the helical twist |
|bayesian_opt_dz_start| the starting value of the helical rise |
|bayesian_opt_dz_end| the ending value of the helical rise |
|bayesian_opt_dz_num| the number of steps for helical rise |
|bayesian_opt_t_num| The number of steps for the parameter t |
|bayesian_opt_1d_search_margin| The margin used for the 1-dimensional search |
|bayesian_opt_1d_search_execute_1| Whether or not to execute the first stage for the helical twist1 |
|bayesian_opt_1d_search_execute_2| Whether or not to execute the first stage for the helical twist2 |
|bayesian_opt_1d_search_execute_3| Whether or not to execute the second stage |
|bayesian_opt_1d_search_load_1| Whether or not to load the data from the first stage for the helical twist1 |
|bayesian_opt_1d_search_load_2| Whether or not to load the data from the first stage for the helical twist2 |
|bayesian_opt_1d_search_load_3| Whether or not to load the data from the second stage |
|bayesian_opt_1d_search_load_folder1| The folder location for the first stage for the helical twist1 |
|bayesian_opt_1d_search_load_folder2| The folder location for the first stage for the helical twist2 |
|bayesian_opt_1d_search_load_folder3| The folder location for the second stage |
|bayesian_opt_2d_search_load| Whether or not to load the data from the 2-dimensional search |
|bayesian_opt_2d_search_load_folder| The folder location for the 2-dimensional search data  |

## Preprocessing
In order to perform byopt-helix, preprocessing is necessary by Relion. 
This includes motion correction, particle picking, and 2D classification.

## Requirements

* Python 3.5
* gpyopt 1.2.5
* Relion 3.0.5

## OS
* Ubuntu 18.04.2 LTS

## License
This software is released under the MIT license. See LICENSE for more information.