import numpy as np
import math
import matplotlib.pyplot as plt
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
import os.path as osp
import sys
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

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, time_range, read_yml, print_dict_summary
from ingest.retrieve_raws_api import get_stations
from ingest.retrieve_raws_stash import get_file_paths

# Read RAWS Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
params_models = read_yml(osp.join(CONFIG_DIR, "params_models.yaml"))
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))

# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})



# Climatology Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

clim_params = Dict(params_models["climatology"])

def time_to_climtimes(t, nyears = clim_params.nyears, ndays=clim_params.ndays):
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

def build_climatology(start, end, bbox, nyears=clim_params.nyears, ndays=clim_params.ndays, min_years = clim_params.min_years):
    
    # Helper Functions
    # def climtimes_df(start, end):
    #     times = time_range(start, end)
    #     clim_df = pd.DataFrame([time_to_climtimes(t) for t in times]).transpose()
    #     return clim_df
        
    def read_st_fm(path, st):
        df = pd.read_pickle(path)  # Read the pickle file
        result = df[df['STID'] == st]  # Filter rows for the specific stid
        
        # If `stid` is missing, add a row with NaNs for other columns
        if result.empty:
            # result = pd.DataFrame({"STID": [st], "datetime": [pd.NA], "fm10": [pd.NA]})
            result = pd.DataFrame({"STID": [st], "datetime": [pd.NaT], "fm10": [float("nan")]})
        return result
    
    def get_fm_data(st, file_paths):
        df_list = []
    
        for i, col in enumerate(file_paths.columns):
            dfi = pd.concat(
                file_paths[col].apply(lambda path: read_st_fm(path, st)).tolist(),
                ignore_index=True
            )
            # Keep only relevant columns and rename `fm10` to `fm_<i>`
            dfi = dfi[["fm10"]].rename(columns={"fm10": i})
            
            # Merge the dataframe on `STID` and `datetime` (by columns)
            df_list.append(dfi)
    
        combined_df = pd.concat(df_list, axis=1)
        return combined_df    
    def count_years(values_df, times_df):
        """
        Based on years in times_df, count number of non-nan values per year in values_df. 
        Result should be a count of the number of years of data with non-nan
        """
        counts = {
            col: times_df[values_df[col].notna()][col].nunique()
            for col in values_df.columns
        }
        counts = pd.Series(counts)
        return counts        

    # Retrieve data
    ## Note, many station IDs will be empty, the list of stids was for the entire bbox region in history
    print(f"Retrieving climatology data from {start} to {end}")
    print("Params for Climatology:")
    print(f"    Number of years to look back: {nyears}")
    print(f"    Number of days to bracked target hour: {ndays}")
    print(f"    Required number of years of data: {min_years}")

    # Get stations in bbox
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]] # Synoptic uses different bbox order
    stids = get_stations(bbox_reordered)["stid"].to_numpy()    
    
    # Calculate times
    times = time_range(start, end)
    clim_df = pd.DataFrame([time_to_climtimes(t) for t in times]).transpose() 
    clim_df.columns = times
    climyears = clim_df.map(lambda x: x.year)

    # Get Stash File Paths 
    file_paths = {col: get_file_paths(clim_df[col]) for col in clim_df.columns}
    file_paths = pd.DataFrame.from_dict(file_paths, orient='index').transpose()    
    
    clim_dict = {}
    for st in stids:
        df = get_fm_data(st, file_paths)
        df.columns = times
        clim_dict[st]={
            "climatology_data": df,
            "queried_times": time_range(start, end)
        }
    
    for st in clim_dict:
        df = clim_dict[st]["climatology_data"]
        years_count = count_years(df, climyears)
        clim_dict[st]["years_count"] = years_count
        fm = np.where(
            years_count>= 6, 
            df.mean(skipna=True),
            np.nan
        )
        clim_dict[st]["fm_forecast"] = fm          
    
    return clim_dict

def get_climatology_forecasts(clim_dict):
    # Get non-empty data and return df
    print("~"*75)
    sts = [*clim_dict.keys()]
    valid_ids = []
    rows = []
    for st in sts:
        dat = clim_dict[st]["fm_forecast"]
        if np.all(np.isnan(dat)):
            print(f"No forecast generated for station {st}, removing")
        else:
            rows.append(dat)
            valid_ids.append(st)
    
    df = pd.DataFrame(np.vstack(rows), index = valid_ids)
    df.columns = clim_dict[sts[0]]["queried_times"] 
    
    print("~"*75)
    print(f"Climatology forecasts of FMC built for {df.shape[0]} unique RAWS stations")

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

# # Parameters
# r0 = 0.05                                   # threshold rainfall [mm/h]
# rs = 8.0                                    # saturation rain intensity [mm/h]
# Tr = 14.0                                   # time constant for rain wetting model [h]
# S = 250                                     # saturation intensity [dimensionless]
# T = 10.0                                    # time constant for wetting/drying

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



ode_params = Dict(params_models["ode"])

class ODE_FMC:
    def __init__(self, params=ode_params):
            
        # List of required keys
        required_keys = ['spinup_hours',
                         'process_variance',
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
    def run_model_single(self, dat, hours, h2, atm_source = "HRRR"):
        """
        Run ODE fuel moisture model on a single location. 
        
        hours : int
            Total hours to run model

        h2 : int
            Hour to turn off data assimilation and run in forecast mode
        
        atm_source: str
            Typically HRRR. Should be able to do RAWS as QC
        """
        Q = self.Q
        R = self.R
        H = self.H
        
        fm = dat["RAWS"]["fm"].to_numpy().astype(np.float64)
        Ed = dat[atm_source]["Ed"].to_numpy().astype(np.float64)
        Ew = dat[atm_source]["Ew"].to_numpy().astype(np.float64)
        rain = dat[atm_source]["rain"].to_numpy().astype(np.float64)

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

    def run_dict(self, dict0, hours, h2, atm_source="HRRR"):
        """
        Run model defined in run_model_single on a dictionary and return 3d array

        Returns
        --------
        u : ndarray
            state vector 3d array of dims (n_locations, timesteps, 2), where 2 dim response is FMC, Ec
        """
        u = []
        for st in dict0:
            ui = self.run_model_single(dict0[st], hours=hours, h2=h2, atm_source=atm_source)
            u.append(ui.T) # transpose to get dimesion (timesteps, response_dim)

        u = np.stack(u, axis=0)
        
        return u

    def slice_fm_forecasts(self, u, h2):
        """
        Given output of run_model, slice array to get only FMC at forecast hours
        """

        return u[:, h2:, 0:1] # Using 0:1 keeps the dimensions, if just 0 it will drop
    
    def eval(self, u, fm, h2):
        """
        Return RMSE of forecast u versus observed FMC
        """
        assert u.shape == fm.shape, "Arrays must have the same shape."
        # Reshape to 2D: (N * timesteps, features)
        fm2 = fm.reshape(-1, fm.shape[-1])
        u2 = u.reshape(-1, u.shape[-1])
        rmse = np.sqrt(mean_squared_error(u2, fm2))
    
        return rmse

    def run_model(self, dict0, hours, h2, atm_source="HRRR"):
        """
        Put it all together
        """
    
        print(f"Running ODE + Kalman Filter with params:")
        print_dict_summary(self.params)
        
        # Get array of response
        fm_arrays = [dict0[loc]["RAWS"]["fm"].values[h2:, np.newaxis] for loc in dict0]
        fm = np.stack(fm_arrays, axis=0)

        # Get forecasts
        preds = self.run_dict(dict0, hours=hours, h2=h2, atm_source=atm_source)
        m = self.slice_fm_forecasts(preds, h2 = h2)

        rmse = self.eval(m, fm, h2)

        return m, rmse







