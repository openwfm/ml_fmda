# Estimating Fuel Moisture Model Accuracy 

There are two modes for training and prediction:
* Evaluation Runs: used to estimate forecast accuracy. Data uses a train/val/test split to estimate forecast accuracy at out-of-sample locations. Training and forecasting is done in the same scripts. Model prediction is done at RAWS locations. Replications can be done with different random seeds to account for training uncertainty. Baseline models are also run, including ODE+KF, climatology, xgboost.
* Operational Runs: used as final prediction for estimating wildfire risk and simulation initialization. No test set is used, all available data used for training and a small validation set to control early stopping. Forecast accuracy estimate is used from an evaluation run in reporting. Training and forecasting are separate processes. Model weights are saved after training. Model prediction is done as a gridded forecast. See `operational.md` for more info

## Forecast Accuracy Analysis

```
sbatch forecast_analysis_controller.sh forecasts/fmc_forecast_test/ etc/rocky_evaluation.yaml
```

## Hyperparameter Tuning
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

## Climatology Baseline Model

Historical weather averages used as a realistic simple baseline. Give config file with bbox and start/end dates for forecast and destination directory for climatology output.

```
sbatch run_climatology.sh etc/config.yaml
```

```
python src/run_climatology.py etc/config.yaml
```



