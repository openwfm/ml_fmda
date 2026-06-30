# Machine Learning Models of Fuel Moisture Content

Project to forecast fuel moisture content with RNNs. Goal is to improve the FMC forecasts used within WRF-SFIRE

## Overview

The Project has components:
- Retrieve and format data for use with ML models
	* FMC data from RAWS
	* Weather data, from RAWS sensors or from HRRR weather model
	* Geographic predictors from RAWS stations or from HRRR or from LandFire
- Train and forecast with RNNs for the purpose of estimating forecast accuracy
	* Train/val/test split, multiple replications with different random seeds to account for training uncertainty
	* Predict with model at RAWS locations so they can be compared to sensor data
	* Run baseline methods of accuracy comparison: ODE, climatology, xgboost
- Train and forecast with RNNs for operational use
	* Do not use a test set. Use all data and rely on forecast accuracy estimates from before
	* Predict with model on HRRR grid to generate regional forecasts
	* Save models for reuse

Data collection, forecast analysis, and saved models are organized by GACC. This is done for performance reasons. Future research should test whether a single national model or several regional models is more accurate

## Setup

### Conda Environments

Due to stability issues with building conda environments, we break up the environment into components:
- Data retrieval environment: uses SynopticPy for FMC data, Herbie for HRRR model, etc. Requires setting up API tokens
	* Name: `ml_fmda_data`
	* Instructions: `install/data_build.txt`
- Error analysis Modeling environment with CPU TensorFlow: used for forecast analysis. Hundreds of replications of training and testing make GPU build infeasible. Parallelization of training replications with CPUs over SLURM
	* Name: `ml_fmda_models`
	* Instructions: `install/env_model.txt` 
- Operational Forecast environment with GPU Tensorflow: run once 
	* Name: `ml_gpu`
	* Instructions: `install/env_gpu.txt`

### API Access

For building datasets from API sources, set up your `token.json` file in order to access APIs by modifying the template file `tokens.json.initial` using VIM or your preferred text editor. If you don’t have one already, you will need a ![SynopticAPI token](https://synopticdata.com/weatherapi/) 

```
cp tokens.json.initial tokens.json
vi tokens.json
```

This project utilizes a cache of RAWS data, since Synoptic charges for data older than 1 year. The stash is maintained by Angel Farguell. The typical workflow is to receive a packaged tar.gz file and extract to the "data" directory with the following command from the Root project directory:

```
tar -xvzf MesoDB.tar.gz data
```

Finally, if you wish to replicate results from the research associated with this project, run the `setup.py` script which will retrieve certain data and set up certain tests

```
python src/setup.py
```

## Running Models

There are two modes for training and prediction:
* Evaluation Runs: used to estimate forecast accuracy. Data uses a train/val/test split to estimate forecast accuracy at out-of-sample locations. Training and forecasting is done in the same scripts. Model prediction is done at RAWS locations. Replications can be done with different random seeds to account for training uncertainty. Baseline models are also run, including ODE+KF, climatology, xgboost. 

```
sbatch rnn_hyperparam_controller.sh models/rnn_hyperparam_tuning_rocky23_TEST/ etc/rnn_hyperparam_tuning_TEST.yaml/
```

```
sbatch forecast_analysis_controller.sh forecasts/fmc_forecast_test/ etc/rocky_evaluation.yaml
```

* Operational Runs: used as final prediction for estimating wildfire risk and simulation initialization. No test set is used, all available data used for training and a small validation set to control early stopping. Forecast accuracy estimate is used from an evaluation run in reporting. Training and forecasting are separate processes. Model weights are saved after training. Model prediction is done as a gridded forecast.

```
sbatch train_cpu.sh etc/sw_operational.yaml
```

```
sbatch train_cpu_reps.sh etc/sw_operational.yaml
```

~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Data Retrieval Description

Retrieved data is organized by RAWS station. This is not necessarily the most computationally efficient approach, but it makes it easier to organize spatiotemporal cross validation and pointwise deployment of baseline models. Data from various sources, including RAWS, HRRR, and LandFire, and combined into one dictionary object.

Workflow Description:
- Retrieve / Ingest data: 
	- Read and combine using either APIs or stashes of saved data. For access to the RAWS stash or HRRR stashes, just ask jonathon.hirschi@ucdenver.edu
	- Interpolate missing RAWS data to regular 1-hr intervals (filters associated with this are applied later) 
	- This process is intended to get all available data relevant to FMC modeling, and it is not affected by choice of particular model predictors or data filtering hyperparameters
	- Uses metadata files for RAWS and HRRR to specify which variables to retrieve and what to name them

Example for building data dictionaries for 2023 in North Rockies GACC, see rtma_cycler for more info
```
sbatch build_fmda_data.sh etc/nr_evaluation.yaml
```


- Build ML Data:
	- Apply interpolation and constant data filters to identify long stretches of constant or perfectly linear data. This filters broken sensors as well as stretches of data that were interpolated past a reasonable limit
	- Merge data sources into a single tabular set of data. Notes written in a subject called "misc" should maintain info on where the data originally came from, but otherwise from this point on the process "forgets" whether the atmospheric data is HRRR, RAWS, or other

- Define Cross-Validation Parameters:
	- Time periods associated with train/val/test
	- List of train/val/test STIDs

- Build Model Specific Data:
	- ODE Data: get data from built ML data using only test STIDs and adjusting the test time periods to account for model spinup (hyper parameter stored in etc/params_models)
	- "Static" ML Data: custom class that handles scaling and reproducibility checks. observations implicitly assumed independent in time, so no need to maintain timeseries connections within the data
	- RNN Data: custom class that handles scaling and reproducibility checks. Restructures data based on (batch_size, time steps, features)


### Data Filters Description

On retrieving the raw data, extreme values filters are applied to the data where RAWS observations are set to NA if outside physically reasonable range. This is done at the data retrieval step since this is based on lab results and physics related to this project. The extreme value filters are not considered a tunable hyper parameter.

To apply filters related to broken sensors or too long stretches of missing data, data is boken into 72 hour periods. This is stored as a hyperparameter in the data_params.yaml file. This is done for the following reasons:

- We want to filter stretches of RAWS with too much missing data, due to either suspect observations or long stretches of interpolated data. Breaking into 72 hour periods allows for removal of bad stretches of RAWS data without filtering out the entire sensor.
- For the ODE+KF to test forecasting 48 hour periods, a 24 hour spinup period for bias correction parameters to stabilize is conservative but appropriate
- 72 hours is divisible by 12 and 24, which are candidates for the timesteps hyperparameter length for defining samples to the RNNs

Changing this from 72 might lead to errors, particularly if changed to something not divisible by 12



## Building Models

### Climatology

Give config file with bbox and start/end dates for forecast and destination directory for climatology output. 

```
sbatch run_climatology.sh etc/config.yaml
```

```
python src/run_climatology.py etc/config.yaml
```

### RNN

For hyperparameter tuning, it is recommended to run on Alderaan or another computing system with many available cores slurm software.
Steps:

1. Confirm hyperparameter search criteria with config file. `etc/rnn_hyperparam_tuning_config.yaml`

2. Run model architecture tuning

```
sbatch rnn_hyperparam_controller.sh models/rnn_hyperparam_tuning_rocky23_TEST/ data/rocky_fmda/
```

3. Run optimization parameter tuning

```
sbatch rnn_hyperparam_controller2.sh models/rnn_hyperparam_tuning_rocky23_TEST/ 
```

For running the corecast analysis, it is recommened to run on Aleraan or another computing system with many available cores and slurm software

1. Run setup and analysis with controller shell file

```
sbatch forecast_analysis_controller.sh forecast_analysis/rm24 etc/rocky_evaluation.yaml
```


## Troubleshooting 

If you repeatedly get the message to setup a Synoptic token despite following the directions from SynopticPy, check your config file: `~/.config/SynopticPy/config.toml`. If the file starts with the line `[default]`, the config file is being read wrong so delete this line and the token will be read properly.

Data retrieval of HRRR can sometimes be killed by automatic processes. If this happens, it is possible that mal-formed subsets of HRRR files get saved in the default directory that HRRR saves temporary files. The mal-formed files need to be deleted, otherwise the Herbie package will detect the file existence by name and not pull in the correct data. If you get errors like "___". Check for directory /Users/USERNAME/data/hrrr/ and delete corresponding files

## Acknowledgements

This research was partially supported by NASA grants 80NSSC23K1118 and 80NSSC23K1344.

A portion of this work used code generously provided by Brian Blaylock's python packages:

Herbie python package (Version 20xx.x.x) (https://doi.org/10.5281/zenodo.4567540)

SynopticPy Python package (https://github.com/blaylockbk/SynopticPy)


