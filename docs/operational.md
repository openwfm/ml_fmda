# Operational Training and Forecasting with an RNN Fuel Moisture Model

Operational runs are used as the final prediction. Intended applications of these predictions are for estimating wildfire risk and simulation initialization. 

No test set is used; all available data used for training and a small validation set to control early stopping. Forecast accuracy estimate is used from an evaluation run in reporting. Training and forecasting are separate processes. Model weights are saved after training. Model prediction is done as a gridded forecast.

```
sbatch train_cpu.sh etc/sw_operational.yaml 
```

```
sbatch train_cpu_reps.sh etc/sw_operational.yaml
```

## Hindcasts

Model is run in forecast mode, i.e. raw inputs given to model with no fitting of parameters to observed FMC data. But this could be deployed over a historical region, possibly over a region and time that was used to train the model. Thus, fhindcast accuracy might not be fully representative of true out-of-sample forecast accuracy. 

```
sbatch hindcast.sh etc/hindcast_TEST.yaml
```

## Forecasts

Model is deployed in real-time. Supports real-time data retrieval and cyclical prediction.


```
sbatch forecast.sh etc/forecast_TEST.yaml
```
