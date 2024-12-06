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

## Acknowledgements

This research was partially supported by NASA grants 80NSSC23K1118 and 80NSSC23K1344.

A portion of this work used code generously provided by Brian Blaylock's python packages:

Herbie python package (Version 20xx.x.x) (https://doi.org/10.5281/zenodo.4567540)

SynopticPy Python package (https://github.com/blaylockbk/SynopticPy)


