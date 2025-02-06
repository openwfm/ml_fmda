# Machine Learning Models of Fuel Moisture Content

Build, train, and deploy machine learning models of fuel moisture content, including recurrent neural networks.

## Setup

First, build and activate the conda environment:

```
conda env create -f environment.yml
conda activate ml_fmda
```

Next, set up your `token.json` file in order to access APIs by modifying the template file `tokens.json.initial` using VIM or your preferred text editor. If you don’t have one already, you will need a ![SynopticAPI token](https://synopticdata.com/weatherapi/) 

```
cp tokens.json.initial tokens.json
vi tokens.json
…
```

This project utilizes a cache of RAWS data, since Synoptic charges for data older than 1 year. The stash is maintained by Angel Farguell. The typical workflow is to receive a packaged tar.gz file and extract to the "data" directory with the following command from the Root project directory:

```
tar -xvzf MesoDB.tar.gz data
```

Finally, if you wish to replicate results from the research associated with this project, run the `setup.py` script which will retrieve certain data and set up certain tests

```
python src/setup.py
…
```


## Data Retrieval Description

Retrieved data is organized by RAWS station. This is not necessarily the most computationally efficient approach, but it makes it easier to organize spatiotemporal cross validation and pointwise deployment of baseline models. Data from various sources, including RAWS, HRRR, and LandFire, and combined into one dictionary object.

Workflow Description:
- Retrieve / Ingest data: 
	- Read and combine using either APIs or stashes of saved data. For access to the RAWS stash or HRRR stashes, just ask jonathon.hirschi@ucdenver.edu
	- Interpolate missing RAWS data to regular 1-hr intervals (filters associated with this are applied later) 
	- This process is intended to get all available data relevant to FMC modeling, and it is not affected by choice of particular model predictors or data filtering hyperparameters

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

```
python models/run_climatology.py '2024-01-01T00:00:00Z' '2024-12-31T23:00:00Z' '[37,-111,46,-95]' climatology_rocky2024.pkl
```


## Troubleshooting 

If you repeatedly get the message to setup a Synoptic token despite following the directions from SynopticPy, check your config file: `~/.config/SynopticPy/config.toml`. If the file starts with the line `[default]`, the config file is being read wrong so delete this line and the token will be read properly.

Data retrieval of HRRR can sometimes be killed by automatic processes. If this happens, it is possible that mal-formed subsets of HRRR files get saved in the default directory that HRRR saves temporary files. The mal-formed files need to be deleted, otherwise the Herbie package will detect the file existence by name and not pull in the correct data. If you get errors like "___". Check for directory /Users/USERNAME/data/hrrr/ and delete corresponding files

## Acknowledgements

This research was partially supported by NASA grants 80NSSC23K1118 and 80NSSC23K1344.

A portion of this work used code generously provided by Brian Blaylock's python packages:

Herbie python package (Version 20xx.x.x) (https://doi.org/10.5281/zenodo.4567540)

SynopticPy Python package (https://github.com/blaylockbk/SynopticPy)


