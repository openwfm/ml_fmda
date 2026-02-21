# Module to test read and join of RAWS, HRRR


import os.path as osp
import sys
import warnings
import numpy as np

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import str2time, hash_ndarray
from ingest.get_fmda_data import retrieve_fmda_data

start = str2time('2024-01-01T00:00:00Z')
end = str2time('2024-01-01T02:00:00Z')
bbox = [40, -105, 45, -100]


# Expected Reproducible Hashes of Data (As of Jan 28, 2025)
stids = ['BRLW4', 'HSYN1', 'HRSN1', 'SBFN1', 'DOHS2', 'BKFS2', 'CRRS2', 'NMOS2', 'RDCS2', 'RESN1', 'VRFN1', 'PINS2', 'DVLW4', 'WCAS2', 'RHRS2', 'TS485', 'WPKS2', 'CSPS2', 'SDSS2', 'RWES2', 'MTRN1', 'MKVN1', 'TT562', 'TT567', 'SFRS2', 'TT591']
HRSN1_fm_hash = 'fff070206a34f2d599c4a269f0506cc6'
TT591_HRRR_Ed_hash = 'b7cbd20f7e375f81dd59365cdbb85ae3'

if __name__ == '__main__':
    print("Testing getting fmda from various data sources and joining into formatted fmda data")

    d = retrieve_fmda_data(start, end, bbox, raws_source="stash", save_path = osp.join(PROJECT_ROOT, "data/test_data/test_fmda_data.pkl"))
    d_sts = [*d.keys()]
    fm = np.array(d["HRSN1"]["RAWS"]["fm"].to_numpy(), dtype=float)
    hash1 = hash_ndarray(np.round(fm, 8))
    hash2 = hash_ndarray(d["TT591"]["HRRR"]["Ed"].to_numpy())
    
    if d_sts != stids:
        warnings.warn("Returned STIDs don't match expected")

    if hash1 != HRSN1_fm_hash:
        warnings.warn("Hash for FMC data from station HRSN1 doesn't match expected")

    if hash2 != TT591_HRRR_Ed_hash:
        warnings.warn("Hash for Ed data from station TT591 doesn't match expected")
    
    
    if (d_sts == stids) and (hash1 == HRSN1_fm_hash) and (hash2 == TT591_HRRR_Ed_hash):
        print()
        print("Hashes of retrieved data match expected")
        print("TEST PASSED")


    
