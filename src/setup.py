# Project setup code, run after cloning project
# Code will:
    # Retrieve LandFire data
    # ...

import sys
import pickle
import json
import os
import os.path as osp
import zipfile
import rioxarray as rxr
from herbie import Herbie
import synoptic
import subprocess
import warnings

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")
DATA_DIR = osp.join(PROJECT_ROOT, "data")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import retrieve_url, read_yml, str2time
from ingest.get_fmda_data import retrieve_fmda_data

# Get Project Config Info
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
project_paths = read_yml(osp.join(CONFIG_DIR, "paths.yaml"))

if __name__ == '__main__':

    print()
    print("~"*75)
    print(f"Checking Paths from etc/paths.yaml")
    
    # Check for RAWS stash.
    # If doesn't exist, need to unpack tar.gz file,
    # Considering automating this, but too many paths so just returning a warning
    # Process to run: tar -xvzf MesoDB.tar.gz
    if not osp.exists(osp.dirname(project_paths['raws_stash_path'])):
        warnings.warn(f"RAWS stash path not found, {osp.dirname(project_paths['raws_stash_path']=})")

    # Check for csv file used to filter broken FMC sensors
    # File created in git project fmc_data, planning to automate this and roll into the main ml_fmda project 
    if not osp.exists(project_paths["valid_path"]):
        warnings.warn(f"No csv found for data QC checks for RAWS stations, so there will be limited filtering of suspect sensor data")

    # Check for HRRR stash path
    if not osp.exists(project_paths["hrrr_stash_path"]):
        print(f"Creating HRRR stash path at {project_paths['hrrr_stash_path']}")
        print(f"NOTE: ensure that this directory can handle a lot of data")
        print(f"NOTE: this isn't where Herbie will direct temporary files, just where processed files are saved")
        os.mkdir(project_paths['hrrr_stash_path'])

    print()
    print("~"*75)
    print(f"Retrieving LandFire data to target directory: {DATA_DIR}")
    # Elevation Data
    # Check that needed tif file exists, if not retrieve from URL and unzip
    if osp.exists(osp.join(project_paths['landfire_elev_dir'], "Tif", "LC20_Elev_220.tif")):
        print(f"LandFire elevation data already exists at {project_paths['landfire_elev_dir']}")
    else:
        # Elevation Data URL and paths
        elev_url = "https://landfire.gov/data-downloads/US_Topo_2020/LF2020_Elev_220_CONUS.zip"
        elev_zipname = "LF2020_Elev_220_CONUS.zip"
        elev_path = osp.join(project_paths['landfire_elev_dir'], "Tif", "LC20_Elev_220.tif")
        # Get Elevation from URL
        print("    Retrieving LandFire Elevation Data from URL")
        retrieve_url(
            url = elev_url,
            dest_path = osp.join(osp.dirname(project_paths['landfire_elev_dir']), elev_zipname)
        ) 
        if not osp.exists(elev_path):
            print("Unzipping file")
            with zipfile.ZipFile(osp.join(osp.dirname(project_paths['landfire_elev_dir']), elev_zipname), 'r') as zip_ref:
                zip_ref.extractall(osp.dirname(project_paths['landfire_elev_dir'])) 
        else:
            print(f"Unzipped Data Exists at: {elev_path}")
    

    # Configure Synoptic Token with tokens.json
    print()
    print("Setting Mesowest token with synopticpy")
    with open(osp.join(CONFIG_DIR, "tokens.json"), "r") as json_file:
        config = json.load(json_file)   
    meso_token = config["mesowest"]
    command = f"export SYNOPTIC_TOKEN={meso_token}"
    subprocess.run(command, shell=True)

    # Get a HRRR File for Spatial Projection Into
    # hpath = Herbie("2025-01-01", product="prs").download(save_dir = osp.join(DATA_DIR, "test_data"))
    # print(f"Saving Local copy of HRRR grib2 file at: {hpath}")

    print()
    print("~"*75)
    print(f"Setting up tests")
    os.makedirs(osp.join(DATA_DIR, "test_data"), exist_ok=True)

    # Generate a small test set of data
    # Run retireve_fmda_data: gets HRRR data with Herbie and stashes, then retrieves RAWS from stash and joins to HRRR
    start = '2024-01-01T00:00:00Z'
    end = '2024-01-01T06:00:00Z'
    bbox = [40,-110,45,-100]
    retrieve_fmda_data(start, end, bbox, save_path = osp.join(DATA_DIR, "test_data", "test_fmda_dict.pkl"))
    assert osp.exists(osp.join(DATA_DIR, "test_data", "test_fmda_dict.pkl")), f"Test data dictionary not found in {osp.join(DATA_DIR, 'test_data')}. See tests/test_ingest_RAWS.py, test_ingest_HRRR.py, and test_ingest_fmda.py, run directly with python"
    


