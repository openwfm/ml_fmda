# Python Package for Core Functionality

A python package contains core functionality from the project, and is intended to be installed locally and used within the `wrfxpy` project.

The project has its own assocaited `README.md` file, but the important considerations are repeated here.

## Installation
...

From the project root directory, the install command utilizes the `pyproj.toml` file

```
python -m pip install --editable
```

## Core functionality

### Cyclical RNN Prediction Class

Custom class that wraps the standard tensorflow models. Contains support for: 
* Dynamic architecture building with a params file that specifies layers, units and activation 
* Cyclical prediction: recurrent state is saved after each prediction call, and used to resume prediction. Intended to be used within a workflow where large earth science data comes in chunks, e.g. in wrfxpy project where hourly cycles ingest large quantities of NWP and other data sources


## Tests

...

