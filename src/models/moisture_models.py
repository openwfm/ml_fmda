# Functions to fit other moisture models, including ODE+KF and static

import numpy as np
import math
import copy
from abc import ABC, abstractmethod
import xgboost as xg
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import os
import os.path as osp
import sys
import warnings
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed


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
from utils import Dict, time_range, read_yml, print_dict_summary, is_consecutive_hours, read_pkl
from ingest.RAWS import get_stations, get_file_paths
import reproducibility


# Read RAWS Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))

# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})



# Climatology Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def time_to_climtimes(t, nyears = 10, ndays=15):
    """
    Given a time, get the corresponding times that will be used for the climatology method.

    Arguments
        t : datetime
            Reference time for method
        nyears: int
            Number of years to look back for data
        ndays: int
            Number of days to bracket the target time, so t +/- ndays is the goal
    """      

    t_years = time_range(
        start = t - relativedelta(years = nyears),
        end = t,
        freq = pd.DateOffset(years=1)
    )

    # For each year, get range of days plus/minus ndays and append to running times object
    ts = []
    for ti in t_years:
        ti_minus_days = ti - relativedelta(days = ndays)
        ti_plus_days = ti + relativedelta(days = ndays)
        ti_grid = time_range(ti_minus_days, ti_plus_days, freq="1d")
        ts.extend(ti_grid)

    # Trim times based on before input time
    ts = np.array(ts)
    ts = ts[ts < t]
    
    return ts

## Helper functions for climatology

def _load_and_filter_pickle(file_path, sts):
    """Load a pickle file using pd.read_pickle and filter by 'stid' column."""
    try:
        df = pd.read_pickle(file_path)
        df.columns = df.columns.str.lower()
        if isinstance(df, pd.DataFrame) and "stid" in df.columns:
            return df[df["stid"].isin(sts)]  # Filter rows
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return None
def _parallel_load_pickles(file_list, sts, num_workers=8):
    """Parallel loading and filtering, using helper function above."""
    results = Parallel(n_jobs=num_workers, backend="loky")(delayed(_load_and_filter_pickle)(f, sts) for f in file_list)
    return pd.concat([df for df in results if df is not None], ignore_index=True)


def _filter_clim_data(clim_data, clim_times):
    """
    Filters clim_data to include only rows where the 'datetime' column matches 
    any datetime in clim_times based on year, month, day, and hour.
    
    Parameters:
    - clim_data (pd.DataFrame): DataFrame containing 'datetime' column (numpy datetime64).
    - clim_times (np.ndarray): Array of datetime objects to match.

    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    # Convert clim_times to a DataFrame for efficient merging
    clim_times_df = pd.DataFrame({
        "year": [t.year for t in clim_times],
        "month": [t.month for t in clim_times],
        "day": [t.day for t in clim_times],
        "hour": [t.hour for t in clim_times]
    }).drop_duplicates()  # Remove duplicates to speed up filtering

    # Extract the relevant time components from clim_data
    clim_data_filtered = clim_data.assign(
        year=clim_data["datetime"].dt.year,
        month=clim_data["datetime"].dt.month,
        day=clim_data["datetime"].dt.day,
        hour=clim_data["datetime"].dt.hour
    ).merge(clim_times_df, on=["year", "month", "day", "hour"], how="inner")

    return clim_data_filtered.drop(columns=["year", "month", "day", "hour"])

def _mean_fmc_by_stid(filtered_df, min_years):
    """
    Computes the average fm10 grouped by 'stid', but returns NaN if the number 
    of unique years in 'datetime' is less than nyears.

    Parameters:
    - filtered_df (pd.DataFrame): DataFrame containing 'stid', 'datetime', and 'fm10'.
    - nyears (int): Minimum number of unique years required per 'stid'.

    Returns:
    - pd.Series: Averaged fm10 per 'stid' (NaN if unique years < nyears).
    """
    # Extract unique years for each STID
    year_counts = filtered_df.groupby("stid")["datetime"].apply(lambda x: x.dt.year.nunique())

    # Compute fm10 average per STID
    fm10_avg = filtered_df.groupby("stid")["fm10"].mean()

    # Set to NaN where unique years < nyears
    fm10_avg[year_counts < min_years] = np.nan

    return fm10_avg

    
def build_climatology(start, end, bbox, clim_params=None):
    """
    Given time period and spatial domain, get all RAWS fm10 data from
    stash based on params. start and end define the forecast hours. 
    Params includes 
        - nyears: number of years back from forecast time to look for data
        - min_years: required number of unique years with available data for a given time and RAWS
        - ndays: number of days to bracket target forecast hour, so target time plus/minus ndays are collected
    """

    if clim_params is None:
        clim_params = Dict(params_models["climatology"])
    nyears = clim_params.nyears
    ndays = clim_params.ndays
    min_years = clim_params.min_years
    

    # Retrieve data
    ## Note, many station IDs will be empty, the list of stids was for the entire bbox region in history
    print(f"Retrieving climatology data from {start} to {end}")
    print("Params for Climatology:")
    print(f"    Number of years to look back: {nyears}")
    print(f"    Number of days to bracked target hour: {ndays}")
    print(f"    Required number of years of data: {min_years}")
    
    # Get target RAWS stations
    sts_df = get_stations(bbox)
    sts = list(sts_df["stid"])

    # Forecast Times, and needed RAWS file hours based on params
    ftimes = time_range(start, end)
    t0 = ftimes.min() - relativedelta(years=clim_params.nyears) - relativedelta(days = clim_params.ndays)
    t1 = ftimes.max()
    all_times = time_range(t0, t1)
    print(f"Total hours to retrieve for climatology: {len(all_times)}")    
    
    raws_files = get_file_paths(all_times)
    raws_files = [f for f in raws_files if os.path.exists(f)]
    print(f"Existing RAWS Files: {len(raws_files)}")    
    
    # Load data and get forecasts
    clim_data = _parallel_load_pickles(raws_files, sts)

    return clim_data

def calculate_fm_forecasts(ftimes, clim_data, clim_params=None):
    """
    Runs `time_to_climtimes` on each time in `ftimes`, filters `clim_data`,
    computes the average `fm10` per `stid`, and combines results.

    Parameters:
    - ftimes (np.ndarray): Array of datetime objects to process.
    - clim_data (pd.DataFrame): DataFrame containing 'stid', 'datetime', and 'fm10'.
    - clim_params: Object containing `nyears` and `ndays` parameters.

    Returns:
    - pd.DataFrame: Combined results with average fm10 per stid for each ftime.
    """
    if clim_params is None:
        clim_params = Dict(params_models["climatology"])
    
    results = []

    for ftime in ftimes:
        # Generate climtimes for the given ftime
        clim_times = time_to_climtimes(ftime, nyears=clim_params.nyears, ndays=clim_params.ndays)
        
        # Filter clim_data based on clim_times
        filtered_data = _filter_clim_data(clim_data, clim_times)

        # Compute the average fm10 per stid
        fm_forecast = _mean_fmc_by_stid(filtered_data, min_years=clim_params.min_years)

        # Store results with corresponding ftime
        df_result = fm_forecast.reset_index()
        df_result["forecast_time"] = ftime  # Add ftime column
        results.append(df_result)

    # Combine all results into a single DataFrame
    df = pd.concat(results, ignore_index=True)
    df = df.pivot(index="stid", columns="forecast_time", values="fm10")    

    # Filter out all NA
    dropped_stids = df.index[df.isna().all(axis=1)].tolist()
    df = df.dropna(how="all")
    print(f"No Data Found for STIDS: {dropped_stids}")
    print(f"Returning forecasts for: {df.shape[0]} unique STIDs")
    
    return df
    
    
# ODE + Augmented Kalman Filter Code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_decay(m0,E,partials=0,T1=0.1,tlen=1):  
    # Arguments: 
    #   m0          fuel moisture content at start dimensionless, unit (1)
    #   E           fuel moisture eqilibrium (1)
    #   partials=0: return m1 = fuel moisture contents after time tlen (1)
    #           =1: return m1, dm0/dm0 
    #           =2: return m1, dm1/dm0, dm1/dE
    #           =3: return m1, dm1/dm0, dm1/dE dm1/dT1   
    #   T1          1/T, where T is the time constant approaching the equilibrium
    #               default 0.1/hour
    #   tlen        the time interval length, default 1 hour

    exp_t = np.exp(-tlen*T1)                  # compute this subexpression only once
    m1 = E + (m0 - E)*exp_t                   # the solution at end
    if partials==0:
        return m1
    dm1_dm0 = exp_t
    if partials==1:
        return m1, dm1_dm0          # return value and Jacobian
    dm1_dE = 1 - exp_t      
    if partials==2:
        return m1, dm1_dm0, dm1_dE 
    dm1_dT1 = -(m0 - E)*tlen*exp_t            # partial derivative dm1 / dT1
    if partials==3:
        return m1, dm1_dm0, dm1_dE, dm1_dT1       # return value and all partial derivatives wrt m1 and parameters
    raise('Bad arg partials')


def ext_kf(u,P,F,Q=0,d=None,H=None,R=None):
    """
    One step of the extended Kalman filter. 
    If there is no data, only advance in time.
    :param u:   the state vector, shape n
    :param P:   the state covariance, shape (n,n)
    :param F:   the model function, args vector u, returns F(u) and Jacobian J(u)
    :param Q:   the process model noise covariance, shape (n,n)
    :param d:   data vector, shape (m). If none, only advance in time
    :param H:   observation matrix, shape (m,n)
    :param R:   data error covariance, shape (n,n)
    :return ua: the analysis state vector, shape (n)
    :return Pa: the analysis covariance matrix, shape (n,n)
    """
    def d2(a):
        return np.atleast_2d(a) # convert to at least 2d array

    def d1(a):
        return np.atleast_1d(a) # convert to at least 1d array

    # forecast
    uf, J  = F(u)          # advance the model state in time and get the Jacobian
    uf = d1(uf)            # if scalar, make state a 1D array
    J = d2(J)              # if scalar, make jacobian a 2D array
    P = d2(P)              # if scalar, make Jacobian as 2D array
    Pf  = d2(J.T @ P) @ J + Q  # advance the state covariance Pf = J' * P * J + Q
    # analysis
    if d is None or not d.size :  # no data, no analysis
        return uf, Pf
    # K = P H' * inverse(H * P * H' + R) = (inverse(H * P * H' + R)*(H P))'
    H = d2(H)
    HP  = d2(H @ P)            # precompute a part used twice  
    K   = d2(np.linalg.solve( d2(HP @ H.T) + R, HP)).T  # Kalman gain
    # print('H',H)
    # print('K',K)
    res = d1(H @ d1(uf) - d)          # res = H*uf - d
    ua = uf - K @ res # analysis mean uf - K*res
    Pa = Pf - K @ d2(H @ P)        # analysis covariance
    return ua, d2(Pa)

### Define model function with drying, wetting, and rain equilibria

# Parameters
ode_params = Dict(params_models["ode"])
r0 = ode_params["r0"] # threshold rainfall [mm/h]
rs = ode_params["rs"] # saturation rain intensity [mm/h]
Tr = ode_params["Tr"] # time constant for rain wetting model [h]
S = ode_params["S"]   # saturation intensity [dimensionless]
T = ode_params["T"]   # time constant for wetting/drying

def model_moisture(m0,Eqd,Eqw,r,t=None,partials=0,T=10.0,tlen=1.0):
    # arguments:
    # m0         starting fuel moistureb (%s
    # Eqd        drying equilibrium      (%) 
    # Eqw        wetting equilibrium     (%)
    # r          rain intensity          (mm/h)
    # t          time
    # partials = 0, 1, 2
    # returns: same as model_decay
    #   if partials==0: m1 = fuel moisture contents after time 1 hour
    #              ==1: m1, dm1/dm0 
    #              ==2: m1, dm1/dm0, dm1/dE  
    
    
    if r > r0:
        # print('raining')
        E = S
        T1 =  (1.0 - np.exp(- (r - r0) / rs)) / Tr
    elif m0 <= Eqw: 
        # print('wetting')
        E=Eqw
        T1 = 1.0/T
    elif m0 >= Eqd:
        # print('drying')
        E=Eqd
        T1 = 1.0/T
    else: # no change'
        E = m0
        T1=0.0
    exp_t = np.exp(-tlen*T1)
    m1 = E + (m0 - E)*exp_t  
    dm1_dm0 = exp_t
    dm1_dE = 1 - exp_t
 
    if partials==0: 
        return m1
    if partials==1:
        return m1, dm1_dm0
    if partials==2:
        return m1, dm1_dm0, dm1_dE
    raise('bad partials')

def model_augmented(u0,Ed,Ew,r,t):
    # state u is the vector [m,dE] with dE correction to equilibria Ed and Ew at t
    # 
    m0, Ec = u0  # decompose state u0
    # reuse model_moisture(m0,Eqd,Eqw,r,partials=0):
    # arguments:
    # m0         starting fuel moistureb (1)
    # Ed         drying equilibrium      (1) 
    # Ew         wetting equilibrium     (1)
    # r          rain intensity          (mm/h)
    # partials = 0, 1, 2
    # returns: same as model_decay
    #   if partials==0: m1 = fuel moisture contents after time 1 hour
    #              ==1: m1, dm0/dm0 
    #              ==2: m1, dm1/dm0, dm1/dE 
    m1, dm1_dm0, dm1_dE  = model_moisture(m0,Ed + Ec, Ew + Ec, r, t, partials=2)
    u1 = np.array([m1,Ec])   # dE is just copied
    J =  np.array([[dm1_dm0, dm1_dE],
                   [0.     ,     1.]])
    return u1, J


# ### Default Uncertainty Matrices
# Q = np.array([[1e-3, 0.],
#             [0,  1e-3]]) # process noise covariance
# H = np.array([[1., 0.]])  # first component observed
# R = np.array([1e-3]) # data variance

# def run_augmented_kf(dat0,h2=None,hours=None, H=H, Q=Q, R=R):
#     dat = copy.deepcopy(dat0)
    
#     if h2 is None:
#         h2 = int(dat['h2'])
#     if hours is None:
#         hours = int(dat['hours'])
    
#     d = dat['y']
#     feats = dat['features_list']
#     Ed = dat['X'][:,feats.index('Ed')]
#     Ew = dat['X'][:,feats.index('Ew')]
#     rain = dat['X'][:,feats.index('rain')]
    
#     u = np.zeros((2,hours))
#     u[:,0]=[0.1,0.0]       # initialize,background state  
#     P = np.zeros((2,2,hours))
#     P[:,:,0] = np.array([[1e-3, 0.],
#                       [0.,  1e-3]]) # background state covariance

#     for t in range(1,h2):
#       # use lambda construction to pass additional arguments to the model 
#         u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
#                                   lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
#                                   Q,d[t],H=H,R=R)
#       # print('time',t,'data',d[t],'filtered',u[0,t],'Ec',u[1,t])
#     for t in range(h2,hours):
#         u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
#                                   lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
#                                   Q*0.0)
#       # print('time',t,'data',d[t],'forecast',u[0,t],'Ec',u[1,t])
#     return u


class ODE_FMC:
    def __init__(self, params=ode_params):
            
        # List of required keys
        required_keys = ['process_variance',
                         'data_variance',
                         'r0',
                         'rs',
                         'Tr',
                         'S',
                         'T']

        # Validate that all required keys are in params
        missing_keys = [key for key in required_keys if key not in params]
        if missing_keys:
            raise ValueError(f"Missing required keys in params: {missing_keys}")

        # Define params
        self.params = params
        process_variance = np.float_(params['process_variance'])
        self.Q = np.array([[process_variance, 0.],
                           [0., process_variance]])
        self.H = np.array([[1., 0.]]) # observation matrix
        self.R = np.array([np.float_(params['data_variance'])]) # data variance
        self.r0 = params["r0"]
        self.rs = params["rs"]
        self.Tr = params["Tr"]
        self.S = params["S"]
        self.T = params["T"]
        
    def run_model_single(self, dat, hours=72, h2=24):
        """
        Run ODE fuel moisture model on a single location. 
        
        hours : int
            Total hours to run model

        h2 : int
            Hour to turn off data assimilation and run in forecast mode
        
        """
        Q = self.Q
        R = self.R
        H = self.H
        
        fm = dat["data"]["fm"].to_numpy().astype(np.float64)
        Ed = dat["data"]["Ed"].to_numpy().astype(np.float64)
        Ew = dat["data"]["Ew"].to_numpy().astype(np.float64)
        rain = dat["data"]["rain"].to_numpy().astype(np.float64)

        u = np.zeros((2,hours))
        u[:,0]=[0.1,0.0]       # initialize,background state  
        P = np.zeros((2,2,hours))
        P[:,:,0] = np.array([[self.params['process_variance'], 0.],
                      [0.,  self.params['process_variance']]]) # background state covariance        
        
        # Run in spinup mode
        for t in range(1,h2):
          # use lambda construction to pass additional arguments to the model 
            u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                    lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                    Q,d=fm[t],H=H,R=R)

        # Run in forecast mode
        for t in range(h2,hours):
            u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                      lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                      Q*0.0)
          
        return u

    def run_dict(self, dict0, hours=72, h2=24):
        """
        Run model defined in run_model_single on a dictionary and return 3d array

        Returns
        --------
        u : ndarray
            state vector 3d array of dims (n_locations, timesteps, 2), where 2 dim response is FMC, Ec
        """
        u = []
        for st in dict0:
            assert is_consecutive_hours(dict0[st]["times"]), f"Input dictionary for station {st} has non-consecutive times"
            ui = self.run_model_single(dict0[st], hours=hours, h2=h2)
            u.append(ui.T) # transpose to get dimesion (timesteps, response_dim)

        u = np.stack(u, axis=0)
        
        return u

    def slice_fm_forecasts(self, u, h2=24):
        """
        Given output of run_model, slice array to get only FMC at forecast hours
        """

        return u[:, h2:, 0:1] # Using 0:1 keeps the dimensions, if just 0 it will drop
    
    def eval(self, u, fm):
        """
        Return RMSE of forecast u versus observed FMC
        """
        
        assert u.shape == fm.shape, "Arrays must have the same shape."
        # Reshape to 2D: (N * timesteps, features)
        fm2 = fm.reshape(-1, fm.shape[-1])
        u2 = u.reshape(-1, u.shape[-1])
        rmse = np.sqrt(mean_squared_error(u2, fm2))
    
        return rmse

    def run_model(self, dict0, hours=72, h2=24):
        """
        Put it all together
        """
    
        print(f"Running ODE + Kalman Filter with params:")
        print_dict_summary(self.params)
        
        # Get array of response
        fm_arrays = [dict0[loc]["data"]["fm"].values[h2:hours, np.newaxis] for loc in dict0]
        fm = np.stack(fm_arrays, axis=0)

        # Get forecasts
        preds = self.run_dict(dict0, hours=hours, h2=h2)
        m = self.slice_fm_forecasts(preds, h2 = h2)

        rmse = self.eval(m, fm)

        return m, rmse


# Static Models Code
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class MLModel(ABC):
    def __init__(self, params: dict):
        self.params = Dict(params)
        if type(self) is MLModel:
            raise TypeError("MLModel is an abstract class and cannot be instantiated directly")
        super().__init__()

    def _filter_params(self, model_cls):
        """Filters out parameters that are not part of the model constructor."""
        model_params = self.params.copy()
        valid_keys = model_cls.__init__.__code__.co_varnames
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}
        return filtered_params
        
    
    def fit(self, X_train, y_train, weights=None):
        print(f"Fitting {self.params.mod_type} with params {self.params}")
        self.model.fit(X_train, y_train, sample_weight=weights)  

    def predict(self, X):
        print(f"Predicting with {self.params.mod_type}")
        preds = self.model.predict(X)
        return preds
        
    # def eval(self, X_test, y_test):
    #     preds = self.predict(X_test)
    #     rmse = np.sqrt(mean_squared_error(y_test, preds))
    #     # rmse_ros = np.sqrt(mean_squared_error(ros_3wind(y_test), ros_3wind(preds)))
    #     print(f"Test RMSE: {rmse}")
    #     # print(f"Test RMSE (ROS): {rmse_ros}")
    #     return rmse

    def run_model(self, data_dict):
        """
        Wrapper to take custom data class and train & predict test data
        """

        self.fit(data_dict.X_train, data_dict.y_train)
        m = self.predict(data_dict.X_test)
        rmse = np.sqrt(mean_squared_error(m, data_dict.y_test))

        return m, rmse
        


class XGB(MLModel):
    def __init__(self, params: dict = None, random_state=None):
        if params is None:
            params = Dict(params_models["xgb"])
        
        super().__init__(params)
        model_params = self._filter_params(XGBRegressor) 
        if random_state is not None:
            reproducibility.set_seed(random_state)
            model_params.update({"random_state": random_state})

        self.model = XGBRegressor(**model_params)
        self.params['mod_type'] = "XGBoost"

    def predict(self, X):
        print("Predicting with XGB")
        preds = self.model.predict(X)
        return preds


class LM(MLModel):
    def __init__(self, params: dict = None):
        if params is None:
            params = Dict(params_models["lm"])
        
        super().__init__(params)
        model_params = self._filter_params(LinearRegression)
        self.model = LinearRegression(**model_params)
        self.params['mod_type'] = "LinearRegression"


