# Set of Functions to process and format fuel moisture model inputs
# These functions are specific to the particulars of the input data, and may not be generally applicable
# Generally applicable functions should be in utils.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import os.path as osp
import sys
import pickle
import pandas as pd
import reproducibility
import random
import copy
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings


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

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, read_pkl, time_range, str2time, is_consecutive_hours, time_intp
import reproducibility
# import ingest.RAWS as rr
# import ingest.HRRR as ih

# Read Variable Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
hrrr_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "hrrr_metadata.yaml"))



# Data Retrieval Wrappers
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# def subdicts_identical(d1, d2, subdict_keys = ["units", "loc", "misc"]):
#     """
#     Helper function to merge retrieved data dictionaries. Checks that subdicts for metadata are the same
#     """
#     return all(d1.get(k) == d2.get(k) for k in subdict_keys)

def subdicts_identical(d1, d2, subdict_keys=["units", "loc"]):
    """
    Helper function to merge retrieved data dictionaries. Checks that subdicts for metadata are the same.
    Returns a tuple: (boolean, list of non-matching keys)
    """
    mismatched_keys = [k for k in subdict_keys if d1.get(k) != d2.get(k)]
    return len(mismatched_keys) == 0, mismatched_keys


def extend_fmda_dicts(d1, d2, st, subdict_keys=["RAWS", "HRRR", "times"]):
    identical, mismatched_keys = subdicts_identical(d1, d2)
    warning_str = ""
    if not identical:
        # warnings.warn(f"Metadata subdicts not the same for station {st}: {mismatched_keys}", UserWarning)
        warning_str = f"Metadata subdicts not the same for station {st}: {mismatched_keys}"
    merged_dict = {k: d1[k] for k in ["units", "loc", "misc"]} # copy metadata

    for key in subdict_keys:
        if key in ["RAWS", "HRRR"]:  # DataFrames
            merged_dict[key] = (
                pd.concat([d1[key], d2[key]])
                .drop_duplicates(subset="date_time")
                .sort_values("date_time")
                .reset_index(drop=True)
            )
        elif key == "times":  # NumPy datetime array
            merged_dict[key] = np.unique(np.concatenate([d1[key], d2[key]]))

    return merged_dict, warning_str

def combine_fmda_files(input_file_paths, save_path = None, verbose=True):
    """
    Read a list of files retrieved with retrieve_fmda_data and combine data at common stations based on time
    """
    # Read all
    dicts = [read_pkl(path) for path in input_file_paths]
    # Initialize combined dictionary as first dict, then loop over others and merge
    combined_dict = dicts[0]
    
    warning_set = set() # Store unique warnings from before
    
    for i in range(1, len(dicts)):
        di = dicts[i]
        for st in di:
            if st not in combined_dict.keys():
                combined_dict[st] = di[st]
            else:
                merged_dict, warning_str = extend_fmda_dicts(combined_dict[st], di[st], st)
                combined_dict[st] = merged_dict
                if warning_str and warning_str not in warning_set:
                    warnings.warn(warning_str, UserWarning)
                    warning_set.add(warning_str)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(combined_dict, f)
    
    return combined_dict


def flag_lag_stretches(x, threshold, lag = 1):
    """
    Used to itentify stretches of data that have been 
    interpolated a length greater than or equal to given threshold. 
    Used to identify stretches of data that are not trustworthy due to 
    extensive interpolation and thus should be removed from a ML training set.
    """
    lags = np.diff(x, n=lag)
    zero_lag_indices = np.where(lags == 0)[0]
    current_run_length = 1
    for i in range(1, len(zero_lag_indices)):
        if zero_lag_indices[i] == zero_lag_indices[i-1] + 1:
            current_run_length += 1
            if current_run_length > threshold:
                return True
        else:
            current_run_length = 1
    else:
        return False  


def sort_files_by_date(path, full_paths =True):
    files = os.listdir(path)
    sorted_files = sorted(files, key=lambda f: f.split('_')[1].split('.')[0])
    if full_paths:
        sorted_files = [osp.join(path, f) for f in sorted_files]
    return sorted_files
    

def build_ml_data(dict0, 
                  hours = 72, 
                  max_linear_time = 10,
                  atm_source="HRRR", 
                  dtype_mapping = {"float": np.float64, "int": np.int64},
                  save_path = None,
                  verbose=True 
                 ):
    """
    Given input of retrieved fmda data, i.e. the output of combine_fmda_files, merge RAWS and HRRR, and apply filters that flag long stretches of interpolated or constant data
    
    Args:
        - hours: number of hours to chop input data into in order to apply the filters that flag lag stretches of data.
        - max_linear_time: if the set of data defined by hours has any stretches of data that are longer than this time, set to NAN as considered untrustworthy. (either broken sensor or unreasonably long time to interpolate)
        - atm_source: Only HRRR now, but should work with RAWS in the future and maybe something else
        - dtype_mapping: (dict) based on metadata dtype string, what to set the column as

    Returns: 
        - ml_dict: dict
    """
    
    # Setup
    d = copy.deepcopy(dict0)
    print(f"Building ML Data with params: ")
    print(f"    {hours=}")
    print(f"    {max_linear_time=}")
    ml_dict = {}
    
    print(f"Merging atmospheric data from {atm_source}")
    # Merge RAWS and HRRR
    for st in d:
        if verbose:
            print("~"*50)
            print(f"Processing station {st}")
        if atm_source == "HRRR":
            raws = d[st]["RAWS"][["stid", "date_time", "fm", "lat", "lon", "elev"]]
            atm = d[st]["HRRR"]
            # Ensure all names renamed, TODO: handle this more gracefully
            # Different versions of Herbie and xarray return different names for the spatial objects for wind and solar. 
            # Manually checking change here, but need to fix moving forward
            atm.rename(columns={"max_10si": "wind", "sdswrf": "solar"}, inplace=True)

            # Check times match
            assert np.all(raws.date_time.to_numpy() == atm.date_time.to_numpy()), f"date_time column doesn't match from RAWS and HRRR for station {st}"

            # Interpolate any missing HRRR 
            # NOTE: I think there should be no NA, but including to be sure, investigate if this is the case
            times = atm.date_time.to_numpy()
            for var in hrrr_meta:
                if var in atm.columns:
                    v = time_intp(times, atm[var].to_numpy(), times)
                    atm.loc[:, var] = v
           

            # Merge, if repeated names add 
            df = pd.merge(
                raws,
                atm,
                left_on=['date_time', 'stid', 'lat', 'lon'],
                right_on=['date_time', 'point_stid', 'point_latitude', 'point_longitude'],
                suffixes=('', '_hrrr')  # Keep the original name for raws, add '_hrrr' for hrrr
            )
        elif atm_source == "RAWS":
            print("RAWS atmospheric data not tested yet")
            sys.exit(-1)
            # df = d[st]["RAWS"]
    
        # Split into periods
        if verbose:
            print(f"Checking {hours} hour increments for constant/linear")
        df['st_period'] = np.arange(len(df)) // hours
    
        # Apply FMC filters and remove suspect data periods. 
        # If no data remaining, add STID to list to remove 
        flagged = df.groupby('st_period')['fm'].apply(
        lambda period: flag_lag_stretches(period, max_linear_time, lag=2)
    ).pipe(lambda flags: flags[flags].index)
        if verbose:
            if flagged.size > 0:
                print(f"Removing period {flagged} due to linear period of data longer than {max_linear_time}")

        # Filter flagged periods
        df_filtered = df[~df['st_period'].isin(flagged)]
        # Set Column types with metadata
        # Apply dtype conversion to each column
        for col, meta in hrrr_meta.items():
            if col in df_filtered.columns:
                target_dtype = dtype_mapping.get(meta.get("dtype"), None)
                if target_dtype:
                    df_filtered = df_filtered.assign(**{col: df_filtered[col].astype(target_dtype)})   
        # Convert Reponse variable type
        df_filtered = df_filtered.assign(fm=pd.to_numeric(df_filtered["fm"]))
        df_filtered = df_filtered.reset_index(drop=True) # restart index counter
                    
        if df_filtered.shape[0] > 0:
            ml_dict[st] = {
                'data': df_filtered,
                'units': d[st]["units"],
                'loc': d[st]["loc"],
                'misc': d[st]["misc"],
                'times': df_filtered["date_time"].to_numpy()
            }
    
    print()
    print(f"Data remaining for {len(ml_dict.keys())} unique stations")
    
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(ml_dict, f)    
    
    return ml_dict


# Cross Validation Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def cv_time_setup(forecast_start_time, train_hours = 8760, forecast_hours = 48, verbose=True):
    """
    Given forecast start time, calculate train/validation/test periods based on parameters.
    """
    
    # Time that forecast period starts
    if type(forecast_start_time) is str:
        t = str2time(forecast_start_time)
    else:
        t = forecast_start_time
    # Start time, number of desired train hours previous to forecast start time, default 1 year (8760) hours
    tstart = t-relativedelta(hours = train_hours)
    # End time, number of forecast hours into future of forecast start time, default 48 hours
    tend = t+relativedelta(hours = forecast_hours-1)

    train_times = time_range(tstart, t-relativedelta(hours = forecast_hours+1))
    val_times = time_range(t-relativedelta(hours = forecast_hours), t-relativedelta(hours = 1))
    test_times = time_range(t, tend)

    if verbose:
        print(f"Start of forecast period: {t}")
        print(f"Number of Training Hours: {len(train_times)}")
        print(f"Train Period: {train_times.min()} to {train_times.max()}")
        print(f"Number of Validation Hours: {len(val_times)}")
        print(f"Val Period: {val_times.min()} to {val_times.max()}")        
        print(f"Number of Forecast Hours: {len(test_times)}")
        print(f"Forecast Period Period: {test_times.min()} to {test_times.max()}")
    
    return train_times, val_times, test_times


def get_stids_in_timeperiod(dict0, times, all_times=True):
    """
    Based on input times, get list of stids from input dictionary 
    that has data availability for that time period.

    Intended use: for static ML models where any samples can be used without
    maintaining sequences, all_times=False. for recurrent ML models where
    sequences need to be maintained, all_times=True.

    Args
        - dict0: (dict) input FMDA dictionary
        - times: (np.array) array of target date times
        - all_times: (bool) if True, only select stids that have 
        full data coverage for input time. If False, any time present 
        will do
    """

    if all_times:
        # Return STIDs where input times are fully included in data times
        stids_output = [stid for stid, data in dict0.items() if set(times).issubset(set(data["times"]))]
    else:
        # Return STIDS where intersection of input times and data times is nonempty
        stids_output = [stid for stid, data in dict0.items() if set(times) & set(data["times"])]

    # Sort return alphabetically, bc set operations non-reproducible
    return sorted(stids_output)

def cv_space_setup(dict0, val_times, test_times, test_frac = 0.1, verbose=True, random_state=None):
    """
    Split cv based on [train, val, test]. Checks for data availability in test and val sets before
    taking sample of size test_frac from total observations. Remaining stations used for train.
    This allows size of train set to vary, but forces consistency of test and val sets
    
    Returns: tuple of lists, train, test, val
    """
    
    if random_state is not None:
        reproducibility.set_seed(random_state)

    # Define size of test/val
    N_t = int(np.round(len(dict0)*test_frac))

    # Select stations from set with data availability
    # in the test time period
    test_ids = get_stids_in_timeperiod(dict0, test_times, all_times=True)
    random.shuffle(test_ids)
    test_locs = test_ids[:N_t]

    # Excluding test locs, select set with data availability
    # in the val time period
    val_ids = get_stids_in_timeperiod(dict0, val_times, all_times=True)
    val_ids = list(set(val_ids) - set(test_locs))
    val_ids.sort() # Sort alphabetically since set operations don't guarantee reproducibility
    random.shuffle(val_ids)
    val_locs = val_ids[:N_t]

    # Get remaining stations for Train set
    stids = [*dict0.keys()]
    train_locs = list(set(stids) - set(val_locs) - set(test_locs))

    if verbose:
        print(f"Total stations: {len(stids)}")
        print(f"Number of train stations: {len(train_locs)}")
        print(f"Number of val stations: {len(val_locs)}")
        print(f"Number of test stations: {len(test_locs)}")
    
    return train_locs, val_locs, test_locs

# Helper function to filter dataframe on time
def filter_df(df, filter_col, ts):
    return df[df[filter_col].isin(ts)]

def get_sts_and_times(dict0, sts_list, times, data_dict = 'data'):
    """
    Given input retrieved fmda data, return sudictionary based on given stations list and observed data times
    """

    d = copy.deepcopy(dict0)

    # Get stations
    new_dict =  {k: d[k] for k in sts_list}

    # Get times
    for st in new_dict:
        new_dict[st]["times"] = times
        new_dict[st][data_dict] = filter_df(new_dict[st][data_dict], "date_time", times)
        new_dict[st]["times"] = new_dict[st][data_dict].date_time.to_numpy()
 
    return new_dict
    
def sort_train_dict(d):
    """
    Rearrange keys based on number of observations in data subkey. Keys with most observations go first

    Used to make stateful batching easier
    
    NOTE: only intended to apply to train dict, as validation and test dicts were constructed so no missing data
    """
    return dict(sorted(d.items(), key=lambda item: item[1]["data"].shape[0], reverse=True))

def filter_empty_data(input_dict):
    return {k: v for k, v in input_dict.items() if v["data"].shape[0] > 0}

def cv_data_wrap(d, fstart, train_hours, forecast_hours, random_state=None):
    """
    Combines functions above to create train/val/test datasets from input data dictionary and params
    """
    
    # Define CV time periods based on params
    train_times, val_times, test_times = cv_time_setup(fstart, train_hours=train_hours, forecast_hours=forecast_hours)
    # Get CV locations based on size of input data dictionary and calculated CV times
    tr_sts, val_sts, te_sts = cv_space_setup(d, 
                                                        val_times=val_times, 
                                                        test_times=test_times, 
                                                        random_state=random_state)
    # Build train/val/test by getting data at needed locations and times
    train = get_sts_and_times(d, tr_sts, train_times)
    val = get_sts_and_times(d, val_sts, val_times)
    test = get_sts_and_times(d, te_sts, test_times)

    # Drop stations with empty data
    train = filter_empty_data(train)
    val = filter_empty_data(val)
    test = filter_empty_data(test)    
    
    return train, val, test


# Final data creation code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_ode_data(dict0, sts, test_times, spinup=24):
    """
    Wraps previous to include a spinup time in the data pulled for test period. Intended to use with ODE+KF model
    """
    d = copy.deepcopy(dict0)

    # Define Spinup Period
    spinup_times = time_range(
        test_times.min()-relativedelta(hours=spinup),
        test_times.min()-relativedelta(hours=1)
    )

    # Get data for spinup period plus test times
    all_times = time_range(spinup_times.min(), test_times.max())
    ode_data = get_sts_and_times(d, sts, all_times)

    # Drop Stations with less than spinup + forecast hours
    return {k: v for k, v in ode_data.items() if v["data"].shape[0] == 72}    
    # return ode_data

class MLData(ABC):
    """
    Abstract base class for ML Data, providing support for scaling. 
    Scaling performed on training data and applied to val and test.
    """    
    def __init__(self, train, val=None, test=None, scaler="standard", features_list=None, random_state=None):
        if random_state is not None:
            reproducibility.set_seed(random_state)
        
        self._run_checks(train, val, test, scaler)

        if scaler not in {"standard", "minmax"}:
            raise ValueError("scaler must be 'standard' or 'minmax'")
        self.scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()
        self.features_list = features_list if features_list is not None else ["Ed", "Ew", "rain"]
        self.n_features = len(self.features_list)

        # Setup data fiels, e.g. X_train and y_train
        self._setup_data(train, val, test)
        # Assuming that units are all the same as it was checked in a previous step
        self.units = next(iter(train.values()))["units"]
    
    def _run_checks(self, train, val, test, scaler):
        """Validates input types for train, val, test, and scaler."""
        if not isinstance(train, dict):
            raise ValueError("train must be a dictionary")
        if val is not None and not isinstance(val, dict):
            raise ValueError("val must be a dictionary or None")
        if test is not None and not isinstance(test, dict):
            raise ValueError("test must be a dictionary or None")
        if scaler not in {"standard", "minmax"}:
            raise ValueError("scaler must be 'standard' or 'minmax'")
    
    @abstractmethod
    def _setup_data(self, train, val, test):
        """Abstract method to initialize X_train, y_train, X_val, y_val, X_test, y_test"""
        pass

    def _combine_data(self, data_dict):
        """Combines all DataFrames under 'data' keys into a single DataFrame."""
        return pd.concat([v["data"] for v in data_dict.values()], ignore_index=True)  
    
    def scale_data(self, verbose=True):
        """
        Scales the training data using the set scaler.
        NOTE: this converts pandas dataframes into numpy ndarrays.
        Tensorflow requires numpy ndarrays so this is intended behavior

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.

        Returns:
        ---------
        Nothing, modifies in place
        """        

        if not hasattr(self, "X_train"):
            raise AttributeError("No X_train within object. Run train_test_split first. This is to avoid fitting the scaler with prediction data.")
        if verbose:
            print(f"Scaling training data with scaler {self.scaler}, fitting on X_train")

        # Fit scaler on training data
        self.scaler.fit(self.X_train)
        # Transform data using fitted scaler
        self.X_train = self.scaler.transform(self.X_train)
        if hasattr(self, 'X_val'):
            if self.X_val is not None:
                self.X_val = self.scaler.transform(self.X_val)
        if self.X_test is not None:
            self.X_test = self.scaler.transform(self.X_test)    

    def inverse_scale(self, save_changes=False, verbose=True):
        """
        Inversely scales the data to its original form. Either save changes internally,
        or return tuple X_train, X_val, X_test

        Parameters:
        -----------
        return_X : str, optional
            Specifies what data to return after inverse scaling. Default is 'all_hours'.
        save_changes : bool, optional
            If True, updates the internal data with the inversely scaled values. Default is False.
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """        
        if verbose:
            print("Inverse scaling data...")
        X_train = self.scaler.inverse_transform(self.X_train)
        X_val = self.scaler.inverse_transform(self.X_val)
        X_test = self.scaler.inverse_transform(self.X_test)

        if save_changes:
            print("Inverse transformed data saved")
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
        else:
            if verbose:
                print("Inverse scaled, but internal data not changed.")
            return X_train, X_val, X_test    
    
    # def print_hashes(self, attrs_to_check = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']):
    #     """
    #     Prints the hash of specified data attributes. 
    #     NOTE: AS OF FEB 3 2025 this doesn't work. data is saved in pandas and reproducibility to_numpy not guarenteed

    #     Parameters:
    #     -----------
    #     attrs_to_check : list, optional
    #         A list of attribute names to hash and print. Default includes 'X', 'y', and split data.
    #     """
        
    #     for attr in attrs_to_check:
    #         if hasattr(self, attr):
    #             value = getattr(self, attr)
    #             print(f"Hash of {attr}: {hash_ndarray(value)}") 


class StaticMLData(MLData):
    """
    Custom class to handle data scaling and extracting from dictionaries. 
    Static combines all data in train/val/test as independent observations in time. 
    So timeseries are not maintained and a single "sample" is one hour of data
    Inherits from MLData class.
    """    
    def _setup_data(self, train, val, test, y_col="fm", verbose=True):
        """
        Combines all DataFrames under 'data' keys for train, val, and test. 
        Static data does not keep track of timeseries, and throws all instantaneous samples into the same pool
        If train and val are None, still create those names as None objects

        Creates numpy ndarrays X_train, y_train, X_val, y_val, X_test, y_test
        """
        if verbose:
            print(f"Subsetting input data to {self.features_list}")

        
        X_train = self._combine_data(train)
        self.y_train = X_train[y_col].to_numpy()
        self.X_train = X_train[self.features_list].to_numpy()

        self.X_val, self.y_val = (None, None)
        if val:
            X_val = self._combine_data(val)
            self.y_val = X_val[y_col].to_numpy()
            self.X_val = X_val[self.features_list].to_numpy()
    
        self.X_test, self.y_test = (None, None)
        if test:
            X_test = self._combine_data(test)
            self.y_test = X_test[y_col].to_numpy()
            self.X_test = X_test[self.features_list].to_numpy()

        if verbose:
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            if self.X_val is not None:
                print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
            if self.X_test is not None:
                print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
            
  
 

    
