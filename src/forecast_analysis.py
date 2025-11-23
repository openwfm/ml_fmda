# Script to run a single iteration of model forecasting given a forecast period
# Runs models, including ODE, XGBoost, and RNN on given forecast period. Save to input directory
# Forecast periods are assigned out with slurm array

import os.path as osp
import sys
import ast
import numpy as np
import pandas as pd
import pickle
import gc
import h5py
from dateutil.relativedelta import relativedelta

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_pkl, read_yml, str2time, time_range
from models.moisture_rnn import model_grid, optimization_grid, RNNData, RNN_Flexible
import data_funcs
import reproducibility
from models.moisture_static import XGB
from models.moisture_ode import ODEData, ODE_FMC
from models.moisture_rnn import RNN_Flexible, RNNData, scale_3d


# Config and metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(f"Usage: {sys.argv[0]} <task_id> <directory>")
        print("Example: python forecast_analysis.py 5 forecasts/fmc_forecast_test")
        sys.exit(-1)    
    
    # Get replication number from slurm task array, forecast directory from user args
    task_id = int(sys.argv[1])
    f_dir = sys.argv[2]
    print(f"Running forecast analysis replication {task_id}")    

    # Read config files, it is created during setup py script
    fconf = Dict(read_yml(osp.join(f_dir, "forecast_config.yaml")))
    params_models = Dict(read_yml(osp.join(f_dir, "params_models.yaml")))

    # Check if output already exists, exit if so.
    # Allows for running multiple times if process stops for any reason
    # Requires manual deletion of old files if you want to rerun
    out_dir = osp.join(f_dir, 'forecast_outputs')
    out_file = osp.join(out_dir, f"fperiod_output_{task_id}.h5")    
    if osp.exists(out_file):
        print(f"Output for task {task_id} already exists at: {out_file}, exiting")
        #sys.exit(0)

    # Get analysis run configuration
    fstart = str2time(fconf.f_start)
    fend = str2time(fconf.f_end)
    tstart = str2time(fconf.train_start)
    tend = str2time(fconf.train_end)
    fhours = fconf.forecast_hours # Number of hours to run forecast starting from initial state (applies to RNN and ODE, doesn't matter for static XGB which has no memory and no path dependence)

    # Get needed data
    # Split train/val/test, use task_id for random seed
    ml_data = read_pkl(osp.join(f_dir, 'ml_data.pkl'))
    reproducibility.set_seed(task_id)
    train, val, test = data_funcs.cv_data_wrap(ml_data, fstart, fend, tstart, tend, val_hours=fconf.val_hours, test_frac = fconf.space_test_frac, random_state=task_id, all_test_times=False)

    # Handle Forecast Periods
    # Define Forecast start times, 48hr spacing
    forecast_periods = time_range(
        start = fstart,
        end = fend,
        freq = f"{fhours}h"
    )
    test_times = time_range(fstart, fend)

    print("~"*75)
    print(f"Running Forecast Analysis replication number {task_id}")

    # Run Models
    baselines = fconf.baselines
    if baselines is None: baselines = []
    te_sts = [*test.keys()]
    column_types = {
        'preds': np.float64,
        'stid': str,
        'date_time': str,
        'fm': np.float64
    } # Used to construct output dataframes

    ## ML Models
    features_list = fconf.features_list

    # RNN
    # Train once, forecast separate times for each period too acount for initial state
    # Loop over forecast periods and build test data
    # Some stations might have missing data for given forecast test period, need to filter those out
    print('~'*75)
    print('Running RNN')
    params = params_models['rnn']
    params.update({'features_list': features_list})
    dat = RNNData(train, val, test=None, method="random", timesteps=fhours, random_state=None, features_list = params["features_list"]) 
    
    # Flag rain
    if 'rain' in features_list: 
        rain_ind = dat.features_list.index("rain")
        r0=0.5 # threshold rain intensity
        rain_cond = dat.X_train[:, :, rain_ind] >= r0
        rain_flag_train = rain_cond.any(axis=1).astype(int)

    dat.scale_data()
    rnn = RNN_Flexible(params=params)
    rnn.fit(dat.X_train, dat.y_train,
            validation_data=(dat.X_val, dat.y_val),
            batch_size = params["batch_size"],
            epochs = params["epochs"],
            #epochs = 1, 
            verbose_fit = True,
            plot_history=False
           ) 
    rnn_output=pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in column_types.items()}) # initialize empty dataframe
    for ft in forecast_periods:
        ts = time_range(ft, ft+relativedelta(hours = fconf.forecast_hours-1))
        # Extract needed times, remove stations with missing data
        test2 = data_funcs.get_sts_and_times(test, te_sts, ts)
        test2 = {k: v for k, v in test2.items() if v["data"].shape[0] == ts.shape[0]}
        # Small chance of no data for all stations sampled for test set within given period. 
        # NOTE: we get around this by running many replications, systematically searching for 
        # data availability is too inefficient 
        if len(test2) > 1:
            X_test = dat._combine_data(test2, params["features_list"])
            # Flag rain
            rain_cond = X_test[:, :, rain_ind] >= r0
            rain_flag_test = rain_cond.any(axis=1).astype(int)
            # Apply fitted scaler from RNNData to test data
            X_test = scale_3d(X_test, dat.scaler)
            sts = dat._combine_data(test2, ['stid'])
            y_test = dat._combine_data(test2, ['fm'])
            assert (X_test.shape[0] == len(test2)) and (X_test.shape[1]==ts.shape[0]) and (X_test.shape[0:2]==y_test.shape[0:2])
            # Run predictiona and format for output
            m_rnn = rnn.predict(X_test)
            df_temp = pd.DataFrame({'preds': m_rnn.flatten(), 'stid': sts.flatten(), 'date_time':np.tile(ts, m_rnn.shape[0]).astype(str), 'fm': y_test.flatten(), 'sample_number': np.repeat(np.arange(X_test.shape[0]), len(ts)), 'rain_flag': np.repeat(rain_flag_test, len(ts))})
            rnn_output = pd.concat([rnn_output, df_temp], ignore_index=True)


    ## Static XGBoost
    if 'xgb' in baselines:
        print('~'*75)
        print("Running XGB")
        params = params_models['xgb']
        params.update({'features_list': features_list})
        dat = data_funcs.StaticMLData(train, val, test, features_list = features_list)
        dat.scale_data()
        xgb_model = XGB(params=params)
        xgb_model.fit(dat.X_train, dat.y_train)
        m_xgb = xgb_model.predict(dat.X_test) # Shape (n_loc*n_time, )
        ## Format output with columns for time and STID
        ## repeat test times for each unique station
        xgb_output = pd.DataFrame({'preds': m_xgb, 'stid': dat.test_locs, 'date_time': dat.test_times.astype(str), 'fm': dat.y_test})

        # Clear up space
        del dat
        gc.collect()

    # ODE
    # Loop over forecast period, get spinup hours and forecast hours for each station
    # might not be enough data for each test station each time
    if 'ode' in baselines:
        print('~'*75)
        print("Running ODE")
        params = params_models['ode']
        ode = ODE_FMC(params=params)
        ode_output=pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in column_types.items()}) # initialize empty dataframe
       
        for ft in forecast_periods:
            print("~"*50)
            print(f"{ft=}")
            ts = time_range(ft, ft+relativedelta(hours = fconf.forecast_hours-1))
            ode_data = ODEData(ml_data, te_sts, ts, spinup=params['spinup_hours'])
            # Small chance of insufficient data for all stations sampled for test set
            if len(ode_data)>1:
                m_ode, fm = ode.run_model(ode_data, hours=ts.shape[0]+params['spinup_hours'], h2=params['spinup_hours'])
                sts = [*ode_data.keys()]
                df_temp = pd.DataFrame({'preds': m_ode.flatten(), 'stid': np.repeat(sts, m_ode.shape[1]), 'date_time':np.tile(ts, m_ode.shape[0]).astype(str), 'fm': fm.flatten()})
                ode_output = pd.concat([ode_output, df_temp], ignore_index=True)
       
        del ode_data
        gc.collect()


    # Climatology
    ## Was run once for all stations, not with a train/test split
    ## Based on current random test set, get climatology predictions for whole year
    ## Get FM from test data
    if 'clim' in baselines:
        clim_file = fconf.climatology_file
        clim = read_pkl(clim_file)
        clim = clim[clim.index.isin(te_sts)]
        clim = clim.loc[:, clim.columns.isin(test_times)]
        clim = clim.reset_index().melt(id_vars='stid', var_name='date_time', value_name='preds')
        clim = clim.sort_values(by=["stid", "date_time"])
        fm = pd.concat([subdict['data'][["stid", "date_time", "fm"]] for subdict in test.values()], ignore_index=True)
        fm.date_time = fm.date_time.astype(str)
        clim.date_time = clim.date_time.astype(str)
        clim_output = clim.merge(fm, on=["stid", "date_time"], how="left")
        clim_output = clim_output[(~clim_output.preds.isna()) & (~clim_output.fm.isna())]

    # Write output
    # Use same h5 file, separate keys for different models (NOTE: mode w vs a for write/append)
    print(f"Writing forecast output for RNN and baselines {fconf.baselines} to file {out_file}")
    rnn_output.to_hdf(out_file, key="rnn", mode="w")
    
    if 'ode' in baselines: ode_output.to_hdf(out_file, key="ode", mode="a")
    if 'xgb' in baselines: xgb_output.to_hdf(out_file, key="xgb", mode="a")
    if 'clim' in baselines: clim_output.to_hdf(out_file, key="clim", mode="a")



