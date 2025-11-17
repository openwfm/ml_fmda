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

project_paths = Dict(read_yml(osp.join(CONFIG_DIR, "paths.yaml")))

# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

smap_config = {
    'product': "SPL3SMP_E",   # daily, 9 km enhanced soil moisture
    'bbox': (-125, 24.94, -66.93, 49.6) # CONUS, [south_lat, west_lon, north_lat, east_lon]
}


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc>' % sys.argv[0]))
        print("Example: python src/ingest/get_smap_data.py '2023-01-01', '2023-01-02'")
        print("Times should match format: YYYY-MM-DD")
        sys.exit(1)

    start = sys.argv[1]
    end = sys.argv[2]
    smap_stash_path = project_paths["smap_stash_path"]

    if not osp.exists(smap_stash_path):
        print(f"Stash directory doesn't exist: {smap_stash_path}")
        print(f"Update path in `etc/paths.yaml` and manually create directory")
        sys.exit(1)
    
    print(f"Retrieving SMAP data for FMDA stash")
    print(f"    Start Time: {start}")
    print(f"    End Time: {end}")
    print(f"    Stash Path: {smap_stash_path}")

    # Login
    if not osp.exists("~/.netrc"):
        print(f"Auth file ~/.netrc does not exist, create with user and pass")
        sys.exit(1)
    earthaccess.login()

    # Retrieve
    time_range = (start, end)
    granules = earthaccess.search_data(
        short_name=smap_config["product"],
        bounding_box=smap_config["bbox"],
        temporal=time_range
    )
    print(f"Found {len(granules)} granules")
    #files = earthaccess.download(granules, smap_stash_path)


    
    

