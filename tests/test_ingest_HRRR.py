# Module to test read/write of HRRR data


import os.path as osp
import sys
import warnings

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import str2time, hash_ndarray
import ingest.HRRR as ih

start = str2time('2025-01-01T00:00:00Z')
end = str2time('2025-01-01T02:00:00Z')
bbox = [40, -105, 45, -100]

# Dummy RAWS dictionary for testing
raws_dict = {
    "STID1":{
        "loc": {"stid": "STID1", "lat": 42, "lon": -102}
    },
    "STID2":{
        "loc": {"stid": "STID2", "lat": 44, "lon": -104}
    }
}

# Expected Reproducible Hashes of Data (As of Jan 28, 2025)
ew_hash = "9d57d0638ea0208fca245b7e83e1aca9"
rain_hash = "f671162f83b9232ee9f94ce0d8b8d3c2"

if __name__ == '__main__':
    print("Testing Ingest HRRR data")

    hrrr_ds = ih.retrieve_hrrr_api(start, end, bbox)
    hrrr_pts = ih.subset_hrrr2raws(hrrr_ds, raws_dict)
    hrrr_pts = ih.rename_ds(hrrr_pts)

    ew = hrrr_pts.Ew.to_numpy()
    rain = hrrr_pts.rain.to_numpy()
    print(f"Wetting Equil. Hash: {hash_ndarray(ew)}")
    print(f"    Expected Hash: {ew_hash}")
    print(f"Rain Hash: {hash_ndarray(rain)}")
    print(f"    Expected Hash: {rain_hash}")
    
    if hash_ndarray(ew) != ew_hash:
        warnings.warn("Wetting Equil. Hash from retrieved HRRR data doesn't match expected")
    if hash_ndarray(rain) != rain_hash:
        warnings.warn("Rain Hash from retrieved HRRR data doesn't match expected")

    if hash_ndarray(ew) == ew_hash and hash_ndarray(rain) == rain_hash:
        print()
        print("Hashes of retrieved data match expected")
        print("TEST PASSED")
    
