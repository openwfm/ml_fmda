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

## Installation

From the repository root, create and activate an environment with TensorFlow
and the package dependencies, then install the package in editable mode:

```bash
python -m pip install --editable .
```

See `install/gpu_build.txt` for Linux CUDA setup or `install/gpu_mac_build.txt`
for Apple-silicon Metal setup.

## Acknowledgements

This work was partially funded by NASA grants XXX. Computational resources were
provided by the Alderaan computing cluster.

## AI use disclosure

OpenAI Codex and ChatGPT were used as coding assistants and documentation
editors. The author reviewed and directed their use and remains responsible for
the software and documentation.
