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

## Code Directory

### Data Processing

Retrieved data is organized by RAWS station. This is not necessarily the most computationally efficient approach, but it makes it easier to organize spatiotemporal cross validation and pointwise deployment of baseline models.

Retrieved data is boken into 72 hour periods. This is stored as a hyperparameter in the data_params.yaml file. This is done for the following reasons:

- We want to filter stretches of RAWS with too much missing data, due to either suspect observations or long stretches of interpolated data. Breaking into 72 hour periods allows for removal of bad stretches of RAWS data without filtering out the entire sensor.
- For the ODE+KF to test forecasting 48 hour periods, a 24 hour spinup period for bias correction parameters to stabilize is conservative but appropriate
- 72 hours is divisible by 12 and 24, which are candidates for the timesteps hyperparameter length for defining samples to the RNNs

Changing this from 72 might lead to errors, particularly if changed to something not divisible by 12

## Acknowledgements

This research was partially supported by NASA grants 80NSSC23K1118 and 80NSSC23K1344.

A portion of this work used code generously provided by Brian Blaylock's python packages:

Herbie python package (Version 20xx.x.x) (https://doi.org/10.5281/zenodo.4567540)

SynopticPy Python package (https://github.com/blaylockbk/SynopticPy)


