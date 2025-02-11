# Run analysis of model forecasting accuracy

import sys
import pickle
import os.path as osp
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
import models.moisture_models as mm

# Read Metadata and Data Params 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params_data = Dict(read_yml(osp.join(CONFIG_DIR, "params_data.yaml")))



# Define Global Variables for analysis
# TRAIN_HOURS = 8760 # includes validation set hours, which is set to same as forecast hours
TRAIN_HOURS = 72
FORECAST_HOURS = 48

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc> <fmda_file_path> <output_file>' % sys.argv[0]))
        print("Example: python src/run_forecast_analysis.py '2023-02-01T00:00:00Z' '2023-02-10T00:00:00Z' 'data/rocky_fmda' data/forecast_analysis_test.pkl")
        print("bbox format should match rtma_cycler: [latmin, lonmin, latmax, lonmax]")
        sys.exit(-1)

    fstart = str2time(sys.argv[1])
    fend = str2time(sys.argv[2])
    data_path = sys.argv[3]
    out_file = sys.argv[4]

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
    # Format: {data_path}/YYYYMM/fmda_YYYYMMDD.pkl

    print("~"*75)
    print("Defining CV time periods based on earliest and latest forecast times")
    t0 = forecast_periods.min() - relativedelta(hours=TRAIN_HOURS)
    t1 = forecast_periods.max() + relativedelta(hours=FORECAST_HOURS)
    days = time_range(t0, t1, freq="1d")
    print(f"Days of Data Needed: {days.shape[0]}")
    print(f"Earliest Day of Data: {days.min()}")
    print(f"Latest Day of Data: {days.max()}")

    file_paths = [f"{data_path}/{dt.strftime('%Y%m')}/fmda_{dt.strftime('%Y%m%d')}.pkl" for dt in days]
    
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
        print(f"All Needed Data exists in {data_path}, proceeding...")

    # Read and Format Data, get set up for train and test
    print("~"*75)
    data = data_funcs.combine_fmda_files(file_paths)
    ml_dict = data_funcs.build_ml_data(data, hours=params_data.hours, 
                                   max_linear_time = params_data.max_linear_time, 
                                   save_path = osp.join(PROJECT_ROOT, out_file))
    print(f"Total Stations with Data in Time Period: {len(ml_dict)}")

    # 
    reproducibility.set_seed(42) # Set seed once here, don't do in the loops
    for tf in forecast_periods:
        print("~"*75)
        print(f"Running train and forecast for period {tf}")
        # Handle Time Cross Val
        train_times, val_times, test_times = data_funcs.cv_time_setup(tf, 
                                                train_hours=TRAIN_HOURS, forecast_hours=FORECAST_HOURS)
        # Handle Location Cross Val, don't set random state so we get different samples for stations
        tr_sts, val_sts, te_sts = data_funcs.cv_space_setup(ml_dict, 
                                                    val_times=val_times, 
                                                    test_times=test_times, 
                                                    random_state=None)
        train = data_funcs.get_sts_and_times(ml_dict, tr_sts, train_times)
        val = data_funcs.get_sts_and_times(ml_dict, val_sts, val_times)
        test = data_funcs.get_sts_and_times(ml_dict, te_sts, test_times)

        # Run Models
        ## ODE
        print()
        ode_data = data_funcs.get_ode_data(ml_dict, te_sts, test_times)
        ode = mm.ODE_FMC()
        m, errs = ode.run_model(ode_data, hours=72, h2=24)
        print(f"ODE RMSE Over Test Period: {errs}")

        ## Static XGBoost
        print()
        dat = data_funcs.StaticMLData(train, val, test)
        dat.scale_data()
        xgb_model = mm.XGB(mm.xgb_params)
        m, err = xgb_model.run_model(dat)
        print(f"XGBoost RMSE over Test Period: {err}")




    