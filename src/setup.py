# Project setup code, run after cloning project
# Code will:
    # Retrieve LandFire data
    # ...

import sys
import pickle
import json
import os.path as osp
import zipfile
import rioxarray as rxr
from herbie import Herbie
import synoptic
import subprocess

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
from utils import retrieve_url


if __name__ == '__main__':

    print()
    print("~"*75)
    print(f"Retrieving LandFire data to target directory: {DATA_DIR}")
    # Elevation Data
    elev_url = "https://landfire.gov/data-downloads/US_Topo_2020/LF2020_Elev_220_CONUS.zip"
    elev_zipname = "LF2020_Elev_220_CONUS.zip"
    elev_path = osp.join(DATA_DIR, "LF2020_Elev_220_CONUS", "Tif", "LC20_Elev_220.tif")

    # Get Elevation
    print()
    print("~"*50)
    print("Retrieving LandFire Elevation Data")
    retrieve_url(
        url = elev_url,
        dest_path = osp.join(DATA_DIR, elev_zipname)
    )
    
    if not osp.exists(elev_path):
        print("Unzipping file")
        with zipfile.ZipFile(osp.join(DATA_DIR, elev_zipname), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)  
    else:
        print(f"Unzipped Data Exists at: {elev_path}")
    
    # Canopy Cover (CC) Data
    canopy_url = "https://landfire.gov/data-downloads/US_230/LF2022_CC_230_CONUS.zip"
    canopy_zipname = "LF2022_CC_230_CONUS.zip"
    canopy_path = osp.join(DATA_DIR, "LF2022_CC_230_CONUS", "Tif", "LC22_CC_230.tif")


    # Get Canopy
    print()
    print("~"*50)
    print("Retrieving LandFire Canopy Cover Data")
    retrieve_url(
        url = canopy_url,
        dest_path = osp.join(DATA_DIR, canopy_zipname)
    )
    
    if not osp.exists(canopy_path):
        print("Unzipping file")
        with zipfile.ZipFile(osp.join(DATA_DIR, canopy_zipname), 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)      
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
    # hpath = herbie.Herbie("2025-01-01", product="prs").download(save_dir = data_path)
    # print(f"Saving Local copy of HRRR grib2 file at: {hpath}")

    



