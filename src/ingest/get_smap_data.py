# Executable process to retrieve SMAP data, format, and save to stash
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# User inputs a start and end time to this process
# Retrieval config as of Nov 17 2025
#     Level 3, 9km enhanced
#     Earth data product SPL3SMP_E
#     Getting all of Conus
# Need to have auth file ~/.netrc


import numpy as np
import earthaccess
import h5py
import os
import os.path as osp
import sys

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.dirname(osp.normpath(CURRENT_DIR)))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_yml
import SMAP as smap

project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))
smap_meta = Dict(read_yml(osp.join(CONFIG_DIR, "variable_metadata", "smap_metadata.yaml")))


# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc>' % sys.argv[0]))
        print("Example: python src/ingest/get_smap_data.py '2023-01-01' '2023-01-02'")
        print("Times should match format: YYYY-MM-DD")
        sys.exit(1)

    # Get user args
    start = sys.argv[1]
    end = sys.argv[2]

    # Login credentials
    if not osp.exists(osp.expanduser("~/.netrc")):
        print(f"Auth file ~/.netrc does not exist, create with user and pass")
        sys.exit(1)
    earthaccess.login()

    # Get granules, open files
    granules = smap.get_granules(start, end, smap_meta)
    if not granules:
        raise SystemExit("No granules found for the specified product and time range.")
    files = earthaccess.open(granules)

    # Get metadata and projection info
    fname = files[1]
    print(f"Extracting metadata from file: {fname}")
    proj, coords = smap.get_smap_spatial(fname)
    breakpoint()




    
    

