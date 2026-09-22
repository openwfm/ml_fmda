# Data Retrieval and Formatting for ML Model Training

Fuel moisture observations are from RAWS. The core weather input is from the HRRR weather model (v4). Data for model training is organized by RAWS, with HRRR weather data interpolated to that location. Data is organized into nested dictionaries.

Other data sources include SMAP soil moisture, LANDFIRE elevation and other static features, ...

Workflow Description:
- Retrieve / Ingest data:
        - Read and combine using either APIs or stashes of saved data. For access to the RAWS stash or HRRR stashes, just ask jonathon.hirschi@ucdenver.edu
        - Interpolate missing RAWS data to regular 1-hr intervals (filters associated with this are applied later)
        - This process is intended to get all available data relevant to FMC modeling, and it is not affected by choice of particular model predictors or data filtering hyperparameters
        - Uses metadata files for RAWS and HRRR to specify which variables to retrieve and what to name them

The `etc/` directory of config files includes bounding boxes and other setup terms for GACC regions. 

Example for building data dictionaries for 2023 in North Rockies GACC.
```
sbatch build_fmda_data.sh etc/nr_evaluation.yaml
```

## Building ML Data

- Apply interpolation and constant data filters to identify long stretches of constant or perfectly linear data. This filters broken sensors as well as stretches of data that were interpolated past a reasonable limit
- Merge data sources into a single tabular set of data. Notes written in a subject called "misc" should maintain info on where the data originally came from, but otherwise from this point on the process "forgets" whether the atmospheric data is HRRR, RAWS, or other

## Data Filters Description

On retrieving the raw data, extreme values filters are applied to the data where RAWS observations are set to NA if outside physically reasonable range. This is done at the data retrieval step since this is based on lab results and physics related to this project. The extreme value filters are not considered a tunable hyper parameter.

To apply filters related to broken sensors or too long stretches of missing data, data is boken into 72 hour periods. This is stored as a hyperparameter in the data_params.yaml file. This is done for the following reasons:

- We want to filter stretches of RAWS with too much missing data, due to either suspect observations or long stretches of interpolated data. Breaking into 72 hour periods allows for removal of bad stretches of RAWS data without filtering out the entire sensor.
- For the ODE+KF to test forecasting 48 hour periods, a 24 hour spinup period for bias correction parameters to stabilize is conservative but appropriate
- 72 hours is divisible by 12 and 24, which are candidates for the timesteps hyperparameter length for defining samples to the RNNs

Changing this from 72 might lead to errors, particularly if changed to something not divisible by 12

## Retrieving Remote Sensing data

SMAP Level 4 soil moisture, stash the `sm_surface` data with the folling

```
sbatch retrieve_smap.sh '2023-01-01' '2024-12-31'
```

Training with L4 since complete records, but TODO is to test real-time forecast with L2/L3
