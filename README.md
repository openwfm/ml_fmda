# Machine Learning Models of Fuel Moisture Content

Project to forecast fuel moisture content with RNNs. Goal is to improve the FMC forecasts used within WRF-SFIRE

## Overview

The Project is broadly divided into the components:
- Retrieve and format data for use with ML models
	* FMC data from RAWS
	* Weather data, from RAWS sensors or from HRRR weather model
	* Geographic predictors from RAWS stations or from HRRR or from LandFire
	* Remote sensing product (SMAP)
	* See `docs/data.md` for more info
- Train and forecast with RNNs for the purpose of estimating forecast accuracy. 
	* Train/val/test split, multiple replications with different random seeds to account for training uncertainty
	* Predict with model at RAWS locations so they can be compared to sensor data
	* Run baseline methods of accuracy comparison: ODE, climatology, xgboost
	* See `docs/forecast_accuracy.md`
- Train and forecast with RNNs for operational use
	* Do not use a test set. Use all data and rely on forecast accuracy estimates from before
	* Predict with model on HRRR grid to generate regional forecasts
	* Save models for reuse
	* See `docs/operational.md` for more info
- A reusable Python Package, for reusing core functionality
	* Custom RNN classes for training and a separate one for prediction
	* The `RNNTraining` class uses tensorflow functional interface to build models, structure data for training, sets up callbacks, etc. This class is used for training and downstream processes of estimating forecast accuracy, hyperparameter tuning, etc.
	* The `OperationalRNNPredictor` class supports cyclical prediction, where the recurrent state is stored. This allows for real-time prediction as data is ingested in cycles. This class is not compiled and lacks support for training. The intention is to build the model from existing weights from the training class
	* See `docs/package.md` for more info
- A tagged version of the project exists for reproducing important papers. Associated publications include:
	* Core RNN description, training procedure, and forecast accuracy experiment. Published in MDPI, https://doi.org/10.3390/fire9010026
	* PhD thesis on time-warping transfer learning for LSTMs, with application for fuel moisture classes. https://doi.org/10.48550/arXiv.2604.02474
	* Transfer learning paper related to thesis work, submitted to AMS AI for the Earth Systems (in review as of Sept 22 2026). This paper uses the related project `fmc_transfer`, but relies on a stable tagged version of the `ml_fmda` repo
	* See `docs/papers.md` for more info


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


## Acknowledgements

This research was partially supported by NASA grants 80NSSC23K1118 and 80NSSC23K1344.

A portion of this work used code generously provided by Brian Blaylock's python packages:

Herbie python package (Version 20xx.x.x) (https://doi.org/10.5281/zenodo.4567540)

SynopticPy Python package (https://github.com/blaylockbk/SynopticPy)


ChatGPT and Codex were used as a coding assistant, an aid for literature search, ...

