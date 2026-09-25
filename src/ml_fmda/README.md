# ml_fmda

`ml_fmda` provides operational RNN prediction for fuel moisture content. It does not provide a training workflow. `src/ml_fmda/README.md` documents reproduction of the pretrained-model deployment workflows. Training and training-data construction are documented elsewhere in this repository.

This README is located at `src/ml_fmda/README.md` within the `openwfm/ml_fmda` Git repository. The reproducibility notebooks are also included in the package directory at `src/ml_fmda/`, alongside the package source code.

**Reproducibility version**: These instructions and the associated pretrained model artifacts correspond to Git tag v0.1.0.

The intended use is within `wrfxpy`, where forcing data arrives one hour at a time and recurrent state is carried forward between prediction cycles.

For access to pretrained models and any other information, contact the primary author Jonathon Hirschi at `jonathon.hirschi@ucdenver.edu`.

## Core functionality

- `OperationalRNNPredictor` loads pretrained weights and supports cyclical
  prediction by accepting and returning recurrent state.
- `TimeWarpedFuelClassPredictors` builds FM1, FM10, FM100, and FM1000
  operational predictors from pretrained FM10 weights. The non-FM10 models use
  input- and forget-gate bias warps.

The time-warp wrapper expects FM10 weights, model parameters, and fuel-class warp outputs from the `fmc_transfer` project. The warp mapping must provide `(bi_warp, bf_warp)` pairs for `fm1`, `fm100`, and `fm1000`; FM10 uses the unmodified pretrained weights.

## Setup Instructions

## 1. Clone the repository

Clone the `openwfm/ml_fmda` repository and navigate to its root directory. Check out the `main` branch.

```
git clone https://github.com/openwfm/ml_fmda.git
cd ml_fmda
git checkout main
```

## 2. Create the Conda environment

This is a **minimal** conda env. This environment is sufficient to run the pretrained models and reproduce the tutorials in this README; it does not include the data-retrieval tools required to construct training datasets or train models.

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

Other conda environment setups are described in `install/`, including those with support for data retrieval, model training, and GPU deployment. 

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

All tests should pass without errors.

## 6. Basic package usage

Run `tutorial_basic.ipynb`. Successful execution should reproduce the example predictions and figures without errors.


## 7. Testing the Pretrained models

The pretrained model weights and associated scalers/configuration are distributed separately from the source package as a ZIP archive. The archive includes a README and the models from the training and transfer-learning workflows:

```
fm_transfer/
├── README.md
├── fm_target_models/
└── fm10_source_model/
```

Large training-data objects used to construct the training datasets are not included.

After obtaining the archive, extract it to a local directory:

```bash
unzip fm_transfer.zip
```

A small test dataset for the operational prediction tutorial is distributed separately as a ZIP archive. The archive contains:

```
ml_fmda_test_data/
├── features.json
├── lat.npy
├── lon.npy
└── X_gridded.npy
```

Extract it with:

```bash
unzip ml_fmda_test_data.zip
```

Then open `tutorial_operational.ipynb`, and update the following cells to point to the two extracted directories.

```
# Path to pretrained model objects, unzipped from fm_transfer.zip
MODEL_DIR = "/path/to/fm_transfer"
# Path to test input gridded data
DATA_DIR = "/path/to/ml_fmda_test_data"
```

Then run the notebook. Successful execution should reproduce the example predictions and figures without errors.

## Acknowledgements

This work was partially funded by NASA grants XXX. Computational resources were
provided by the Alderaan computing cluster.

## AI use disclosure

OpenAI Codex and ChatGPT were used as coding assistants and documentation
editors. The author reviewed and directed their use and remains responsible for
the software and documentation.
