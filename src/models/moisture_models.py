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
from utils import Dict, time_range, read_yml
from ingest.retrieve_raws_api import get_stations
from ingest.retrieve_raws_stash import get_file_paths

# Read RAWS Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
raws_meta = read_yml(osp.join(CONFIG_DIR, "variable_metadata", "raws_metadata.yaml"))

# Update stash path. We do this here so it works if module called from different locations
raws_meta.update({'raws_stash_path': osp.join(PROJECT_ROOT, raws_meta['raws_stash_path'])})


# Climatology Method
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def time_to_climtimes(t, nyears = 6, ndays=15):
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

def build_climatology(start, end, bbox, nyears=6, ndays=15, required_years = 6):
    
    # Helper Functions
    def climtimes_df(start, end):
        times = time_range(start, end)
        clim_df = pd.DataFrame([time_to_climtimes(t) for t in times]).transpose()
        return clim_df
        
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


    # Get stations in bbox
    bbox_reordered = [bbox[1], bbox[0], bbox[3], bbox[2]] # Synoptic uses different bbox order
    stids = get_stations(bbox_reordered)["stid"].to_numpy()

    # Retrieve data
    ## Note, many station IDs will be empty, the list of stids was for the entire bbox region in history
    print(f"Retrieving climatology data from {start} to {end}, for {len(stids)} possible RAWS")
    print("Params for Climatology:")
    print(f"    Number of years to look back: {nyears}")
    print(f"    Number of days to bracked target hour: {ndays}")
    print(f"    Required number of years of data: {ndays}")

    # Calculate times
    times_df = climtimes_df(start, end)
    climyears = times_df.map(lambda x: x.year)

    # Get Stash File Paths 
    file_paths = {col: get_file_paths(times_df[col]) for col in times_df.columns}
    file_paths = pd.DataFrame.from_dict(file_paths, orient='index').transpose()    
    
    clim_dict = {}
    for st in stids:
        clim_dict[st]={
            "climatology_data": get_fm_data(st, file_paths)
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

# Parameters
r0 = 0.05                                   # threshold rainfall [mm/h]
rs = 8.0                                    # saturation rain intensity [mm/h]
Tr = 14.0                                   # time constant for rain wetting model [h]
S = 250                                     # saturation intensity [dimensionless]
T = 10.0                                    # time constant for wetting/drying

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
    #if t>=933 and t < 940:
    #  print('t,Eqw,Eqd,r,T1,E,m0,m1,dm1_dm0,dm1_dE',
    #        t,Eqw,Eqd,r,T1,E,m0,m1,dm1_dm0,dm1_dE)   
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


### Default Uncertainty Matrices
Q = np.array([[1e-3, 0.],
            [0,  1e-3]]) # process noise covariance
H = np.array([[1., 0.]])  # first component observed
R = np.array([1e-3]) # data variance

def run_augmented_kf(dat0,h2=None,hours=None, H=H, Q=Q, R=R):
    dat = copy.deepcopy(dat0)
    
    if h2 is None:
        h2 = int(dat['h2'])
    if hours is None:
        hours = int(dat['hours'])
    
    d = dat['y']
    feats = dat['features_list']
    Ed = dat['X'][:,feats.index('Ed')]
    Ew = dat['X'][:,feats.index('Ew')]
    rain = dat['X'][:,feats.index('rain')]
    
    u = np.zeros((2,hours))
    u[:,0]=[0.1,0.0]       # initialize,background state  
    P = np.zeros((2,2,hours))
    P[:,:,0] = np.array([[1e-3, 0.],
                      [0.,  1e-3]]) # background state covariance
    # Q = np.array([[1e-3, 0.],
    #             [0,  1e-3]]) # process noise covariance
    # H = np.array([[1., 0.]])  # first component observed
    # R = np.array([1e-3]) # data variance

    for t in range(1,h2):
      # use lambda construction to pass additional arguments to the model 
        u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q,d[t],H=H,R=R)
      # print('time',t,'data',d[t],'filtered',u[0,t],'Ec',u[1,t])
    for t in range(h2,hours):
        u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                  lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t),
                                  Q*0.0)
      # print('time',t,'data',d[t],'forecast',u[0,t],'Ec',u[1,t])
    return u













