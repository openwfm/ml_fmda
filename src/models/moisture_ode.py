# Module to run ODE+KF moisture model

import numpy as np
from sklearn.metrics import mean_squared_error
import os
import copy
import os.path as osp
import sys
import warnings
from dateutil.relativedelta import relativedelta

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, time_range, read_yml, print_dict_summary, is_consecutive_hours, read_pkl
import reproducibility


# Read Metadata
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ode_params = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="ode"))
# Parameters
r0 = ode_params.r0 # threshold rainfall [mm/h]
rs = ode_params.rs # saturation rain intensity [mm/h]
Tr = ode_params.Tr # time constant for rain wetting model [h]
S = ode_params.S   # saturation intensity [dimensionless]
T = ode_params.T   # time constant for wetting/drying



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


def ODEData(dict0, sts, test_times, spinup=24):
    """
    Wraps previous to include a spinup time in the data pulled for test period. Intended to use with ODE+KF model
    """
    from data_funcs import get_sts_and_times

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
    return {k: v for k, v in ode_data.items() if v["data"].shape[0] == len(all_times)}
    # return ode_data


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
        process_variance = np.float64(params['process_variance'])
        self.Q = np.array([[process_variance, 0.],
                           [0., process_variance]])
        self.H = np.array([[1., 0.]]) # observation matrix
        self.R = np.array([np.float64(params['data_variance'])]) # data variance
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
        Return MSE of forecast u versus observed FMC
        """
        
        assert u.shape == fm.shape, "Arrays must have the same shape."

        # Overall MSE
        # Reshape to 2D: (N * timesteps, features)
        fm2 = fm.reshape(-1, fm.shape[-1])
        u2 = u.reshape(-1, u.shape[-1])
        mse = mean_squared_error(u2, fm2)

        # Per-loc MSE
        batch_mse = np.array([
            mean_squared_error(fm[i].reshape(-1), u[i].reshape(-1))
            for i in range(fm.shape[0])
        ])        

        # Set up error return dict, support for multiple metrics
        errs = {
            'mse': mse,
            'loc_mse': batch_mse
        }
    
        return errs

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

        # errs = self.eval(m, fm)

        return m, fm

if __name__ == '__main__':

    print("Imports successful, no executable code")
