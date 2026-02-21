import sys
import pickle
import os
import os.path as osp
import pandas as pd
from joblib import cpu_count
import yaml

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import parse_bbox, time_range, Dict, read_yml
from models.moisture_climatology import build_climatology, calculate_fm_forecasts

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <config_file>' % sys.argv[0]))
        print("Config file should define a climatology file and directory")
        print("Example: python src/run_climatology etc/forecast_analysis_TEST.yaml")
        sys.exit(-1)

    # Setup
    fconf = read_yml(sys.argv[1])   
    out_file = fconf["climatology_file"]
    out_dir = osp.dirname(out_file)
    os.makedirs(out_dir, exist_ok=True)

    # Write copy of config file to target directory
    with open(osp.join(out_dir, "forecast_config.yaml"), 'w') as f:
        yaml.dump(fconf, f, default_flow_style=False, sort_keys=False)
    fconf = Dict(fconf)    
    start = fconf.f_start
    end = fconf.f_end
    bbox = parse_bbox(fconf.bbox)   
   
    # Read needed data from stash
    if not osp.exists(out_file):
        n_cores = cpu_count() # -1 #if still using system for other tasks
        clim_data = build_climatology(
            start,
            end,
            bbox,
            n_workers = n_cores
        )
        # Write bulk data
        print(f"Writing bulk data to {osp.join(out_dir, 'raw.pkl')}")
        with open(osp.join(out_dir, 'raw.pkl'), 'wb') as handle:
            pickle.dump(clim_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Climatology file {out_file} already exists, reading that file. To rerun, manually delete")
        clim_data = pd.read_pickle(out_file)


    # Get forecasts and save
    ftimes = time_range(start, end) # forecast hours    
    clim_forecasts = calculate_fm_forecasts(ftimes, clim_data) # calc climatology forecasts

    with open(out_file, 'wb') as handle:
        pickle.dump(clim_forecasts, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  
    
    
