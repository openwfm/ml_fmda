# Module to test read/write of HRRR data


import os.path as osp
import os
import sys
import warnings
import numpy as np

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
import ingest.RAWS as rr

bbox = [40, -105, 45, -100]

# Expected Reproducible Hashes of Data (As of Jan 28, 2025)
stids = ['BRLW4', 'HSYN1', 'HRSN1', 'SBFN1', 'DOHS2', 'BKFS2', 'CRRS2', 'NMOS2', 'RDCS2', 'RESN1', 'VRFN1', 'PINS2', 'DVLW4', 'WCAS2', 'RHRS2', 'TS485', 'WPKS2', 'CSPS2', 'SDSS2', 'RWES2', 'MTRN1', 'MKVN1', 'TT562', 'TT567', 'SFRS2', 'TT591']
BRLW4_hash = 'bb5f2ee743f31041253115de43227c99'

if __name__ == '__main__':
    
    start = str2time('2024-01-01T00:00:00Z')
    end = str2time('2024-01-01T02:00:00Z')
    raws_dict = rr.build_raws_dict_stash(start, end, bbox)
    x = np.array(raws_dict["BRLW4"]["RAWS"]["fm"].to_numpy(), dtype=float)
    x = np.round(x, 8)

    
    if [*raws_dict.keys()] != stids:
        warnings.warn("Returned STIDs don't match expected")
    if hash_ndarray(x) != BRLW4_hash:
        warnings.warn("Returned RAWS data for st BRLW4 doesn't match expected")
    
    if [*raws_dict.keys()] == stids and hash_ndarray(x) == BRLW4_hash:
        print()
        print("Hashes of retrieved data match expected")
        print("TEST PASSED")
    