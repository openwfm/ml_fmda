# ml_fmda

`ml_fmda` provides operational RNN prediction for fuel moisture content. It
does not provide a training workflow. training is described in the parent ml_fmda git project, with support for data retireval and validation of forecast accuracy. This package focuses on deploying pretrained models 

The intended use is within `wrfxpy`,
where forcing data arrives one hour at a time and recurrent state is carried
forward between prediction cycles.

## Core functionality

- `OperationalRNNPredictor` loads pretrained weights and supports cyclical
  prediction by accepting and returning recurrent state.
- `TimeWarpedFuelClassPredictors` builds FM1, FM10, FM100, and FM1000
  operational predictors from pretrained FM10 weights. The non-FM10 models use
  input- and forget-gate bias warps.

The time-warp wrapper expects FM10 weights, model parameters, and fuel-class
warp outputs from the `fmc_transfer` project. The warp mapping must provide
`(bi_warp, bf_warp)` pairs for `fm1`, `fm100`, and `fm1000`; FM10 uses the
unmodified pretrained weights.

## Setup Instructions

## 1. Clone the repository

Clone the `openwfm/ml_fmda` repository and navigate to its root directory.

## 2. Create the Conda environment

This is a **minimal** conda env. Core software tools for running the model. Does NOT include spatial (xarray/netcdf4), does not include data API packages

```
conda create -n ml_fmda_model -c conda-forge python=3.11 pip
conda activate ml_fmda_model

conda install -c conda-forge \
    "numpy>=1.24,<1.27" \
    pandas matplotlib scikit-learn \
    jupyter jupyterlab xgboost \
    "tables>=3.8"

pip install "tensorflow==2.16.1"
conda install pytest
```


## 3. Install the package

From the project root:

```
pip install -e .
```

## 4. Verify the installation

```
python -c "import numpy, pandas, sklearn, xgboost, tables, tensorflow; print('numpy', numpy.__version__); print('tensorflow', tensorflow.__version__)"
```

## 5. Run the tests

From project root:

```
pytest tests/test_rnn.py
```

## 6. Basic usage

...


## Acknowledgements

This work was partially funded by NASA grants XXX. Computational resources were
provided by the Alderaan computing cluster.

## AI use disclosure

OpenAI Codex and ChatGPT were used as coding assistants and documentation
editors. The author reviewed and directed their use and remains responsible for
the software and documentation.
