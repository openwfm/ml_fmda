import sys
import pickle
import os.path as osp
import pandas as pd
from joblib import cpu_count


# import multiprocessing
# multiprocessing.set_start_method("spawn", force=True) # Due to warning about 'fork' on linux and mac



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
from utils import parse_bbox, time_range
from models.moisture_models import build_climatology, calculate_fm_forecasts

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc> <bbox> <output_file>' % sys.argv[0]))
        print("Example: python src/run_climatology '2023-06-01T00:00:00Z' '2023-06-01T05:00:00Z' '[37,-105,39,-103]' data/test_climatology")
        print("bbox format should match rtma_cycler: [latmin, lonmin, latmax, lonmax]")
        sys.exit(-1)

    # Setup
    start = sys.argv[1]
    end = sys.argv[2]
    bbox = parse_bbox(sys.argv[3])
    out_file = sys.argv[4]

    # Read needed data from stash
    if not osp.exists(f"{out_file}_data.pkl"):
        n_cores = cpu_count() -1 # remove 1 to keep system usable
        print(f"Running Climatology Read with {n_cores} cores")
        clim_data = build_climatology(
            start,
            end,
            bbox,
            n_workers = n_cores
        )
        # Write bulk data
        with open(f"{out_file}_data.pkl", 'wb') as handle:
            pickle.dump(clim_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(f"Climatology file {out_file}_data.pkl already exists, reading that file. To rerun, manually delete")
        clim_data = pd.read_pickle(f"{out_file}_data.pkl")


    # Get forecasts and save
    ftimes = time_range(start, end) # forecast hours    
    clim_forecasts = calculate_fm_forecasts(ftimes, clim_data) # calc climatology forecasts

    with open(f"{out_file}_forecasts.pkl", 'wb') as handle:
        pickle.dump(clim_forecasts, handle, protocol=pickle.HIGHEST_PROTOCOL)    
  
    
    