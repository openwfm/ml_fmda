# Run analysis of model forecasting accuracy

import sys
import pickle
import os.path as osp
import os
from dateutil.relativedelta import relativedelta

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## We do this so the module can be imported from different locations
CURRENT_DIR = osp.abspath(__file__)
while osp.basename(CURRENT_DIR) != "ml_fmda":
    CURRENT_DIR = osp.dirname(CURRENT_DIR)
PROJECT_ROOT = CURRENT_DIR
CODE_DIR = osp.join(PROJECT_ROOT, "src")
sys.path.append(CODE_DIR)
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")
DATA_DIR = osp.join(PROJECT_ROOT, "data")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict, str2time, time_range
import data_funcs
import reproducibility
from models.moisture_static import XGB
from models.moisture_ode import ODE_FMC
from models.moisture_rnn import RNN_Flexible, RNNData

# Config and Params 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))
params_models = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml")))
params_models['rnn'].update({'timesteps': None}) # For flexible training and prediction
n_features = len(params_data['features_list'])


# Define Global Variables for analysis
# TRAIN_HOURS = 8760 # includes validation set hours, which is set to same as forecast hours
TRAIN_HOURS = 144
FORECAST_HOURS = 48




if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc> <fmda_file_path> <output_file>' % sys.argv[0]))
        print("Example: python src/run_forecast_analysis.py '2023-02-01T00:00:00Z' '2023-02-10T00:00:00Z' 'data/rocky_fmda' outputs/forecast_analysis_test")
        print("bbox format should match rtma_cycler: [latmin, lonmin, latmax, lonmax]")
        sys.exit(-1)

    fstart = str2time(sys.argv[1])
    fend = str2time(sys.argv[2])
    data_dir = sys.argv[3]
    out_dir = sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)

    # Handle Forecast Periods
    # Define Forecast start times, 48hr spacing
    forecast_periods = time_range(
        start = fstart,
        end = fend,
        freq = "2d"
    )
    
    print("~"*75)
    print(f"Running Forecast Analysis from {fstart} to {fend}")
    print(f"Analysis Params: ")
    print(f"    {FORECAST_HOURS=}")
    print(f"    {TRAIN_HOURS=}")
    print(f"Total Forecast Periods: {forecast_periods.shape[0]}")
    print(f"Earliest forecast start: {forecast_periods.min()}")
    print(f"Latest forecast start: {forecast_periods.max()}")


    # Get FMDA Data Paths based on CV Scheme
    # 48hr test periods, 1 year train (incl 48 hour validation)
    # Assuming structure of pickle files for 1 day of some bbox
    # Format: {data_dir}/YYYYMM/fmda_YYYYMMDD.pkl

    print("~"*75)
    print("Defining CV time periods based on earliest and latest forecast times")
    t0 = forecast_periods.min() - relativedelta(hours=TRAIN_HOURS)
    t1 = forecast_periods.max() + relativedelta(hours=FORECAST_HOURS)
    days = time_range(t0, t1, freq="1d")
    print(f"Days of Data Needed: {days.shape[0]}")
    print(f"Earliest Day of Data: {days.min()}")
    print(f"Latest Day of Data: {days.max()}")

    file_paths = [f"{data_dir}/{dt.strftime('%Y%m')}/fmda_{dt.strftime('%Y%m%d')}.pkl" for dt in days]
    
    all_exist = all(osp.exists(path) for path in file_paths)
    # For now, hard exit if not all data exists. Maybe relax in the future
    if not all_exist:
        print(f"Not all needed file paths exist for target analysis. Exiting...")
        missing_paths = [path for path in file_paths if not osp.exists(path)]
        print("Missing files:")
        for path in missing_paths:
            print(path)
        sys.exit(-1)
    else:
        print(f"All Needed Data exists in {data_dir}, proceeding...")

    # Read and Format Data, get set up for train and test
    print("~"*75)
    data = data_funcs.combine_fmda_files(file_paths)
    ml_dict = data_funcs.build_ml_data(data, hours=params_data.hours, 
                                   max_linear_time = params_data.max_linear_time, 
                                   save_path = osp.join(PROJECT_ROOT, out_dir, "ml_data.pkl"), verbose=False)
    print(f"Total Stations with Data in Time Period: {len(ml_dict)}")

    # 
    reproducibility.set_seed(42) # Set seed once here, don't do in the loops
    for ft in forecast_periods:
        print("~"*75)

        out_file = osp.join(PROJECT_ROOT, out_dir, f"{ft.strftime('%Y%m%d_%H')}.pkl")
        if osp.exists(out_file):
            print("Forecast output already exists, skipping to next period")
        else:
            print(f"Running train and forecast for period {ft}")
            # Handle Time Cross Val
            train, val, test = data_funcs.cv_data_wrap(ml_dict,
                                        ft, train_hours=TRAIN_HOURS,forecast_hours=FORECAST_HOURS)
    
            # Run Models
            # ODE
            print()
            te_sts = [*test.keys()]
            test_times = test[te_sts[0]]["times"]
            ode_data = data_funcs.get_ode_data(ml_dict, te_sts, test_times)
            ode = ODE_FMC()
            m, errs_ode = ode.run_model(ode_data, hours=72, h2=24)
            print(f"ODE Test RMSE: {errs_ode}")
    
            ## Static XGBoost
            print()
            dat = data_funcs.StaticMLData(train, val, test)
            dat.scale_data()
            xgb_model = XGB()
            xgb_model.fit(dat.X_train, dat.y_train)
            errs_xgb = xgb_model.test_eval(dat.X_test, dat.y_test, verbose=False)
            print(f"XGBoost Test RMSE: {errs_xgb['rmse']}")

            # RNN
            dat = RNNData(train, val, test, method="random", random_state=None)
            dat.scale_data()
            rnn = RNN_Flexible(n_features=dat.n_features,params=params_models['rnn'])
            rnn.fit(dat.X_train, dat.y_train, 
                    validation_data=(dat.X_val, dat.y_val),
                    batch_size = params_models['rnn']["batch_size"],
                    epochs = 3,
                    verbose_fit = True,
                    plot_history=False
                   )
            errs_rnn = rnn.test_eval(dat.X_test, dat.y_test, verbose=False)
            print(f"RNN Test RMSE: {errs_rnn['rmse']}")

            # Write output for forecast period
            err_dict_output = {
                'ODE': errs_ode,
                'XGB': errs_xgb,
                'RNN': errs_rnn
            }
            print(f"Writing Output: {out_file}")
            with open(out_file, 'wb') as handle:
                pickle.dump(err_dict_output, handle, protocol=pickle.HIGHEST_PROTOCOL)  
            




    