# Module to test read and join of RAWS, HRRR


import os.path as osp
import sys
import warnings

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
from utils import str2time, hash_ndarray
from ingest.get_project_data import retrieve_fmda_data

start = str2time('2024-01-01T00:00:00Z')
end = str2time('2024-01-01T02:00:00Z')
bbox = [40, -105, 45, -100]

# Dummy RAWS dictionary for testing
# raws_dict = {
#     "STID1":{
#         "loc": {"stid": "STID1", "lat": 42, "lon": -102}
#     },
#     "STID2":{
#         "loc": {"stid": "STID2", "lat": 44, "lon": -104}
#     }
# }

# # Expected Reproducible Hashes of Data (As of Jan 28, 2025)
# ew_hash = "9d57d0638ea0208fca245b7e83e1aca9"
# rain_hash = "f671162f83b9232ee9f94ce0d8b8d3c2"

if __name__ == '__main__':
    print("Testing getting fmda from various data sources and joining into formatted fmda data")

    d = retrieve_fmda_data(start, end, bbox, raws_source="stash")
    

    # ew = hrrr_pts.Ew.to_numpy()
    # rain = hrrr_pts.rain.to_numpy()
    # print(f"Wetting Equil. Hash: {hash_ndarray(ew)}")
    # print(f"    Expected Hash: {ew_hash}")
    # print(f"Rain Hash: {hash_ndarray(rain)}")
    # print(f"    Expected Hash: {rain_hash}")
    
    # if hash_ndarray(ew) != ew_hash:
    #     warnings.warn("Wetting Equil. Hash from retrieved HRRR data doesn't match expected")
    # if hash_ndarray(rain) != rain_hash:
    #     warnings.warn("Rain Hash from retrieved HRRR data doesn't match expected")

    # if hash_ndarray(ew) == ew_hash and hash_ndarray(rain) == rain_hash:
    #     print()
    #     print("Hashes of retrieved data match expected")
    #     print("TEST PASSED")
    