# Module to test read/write of HRRR data
# NOTE: as of Sept 18, 2025 there are extra stations in the query: SDSS2. TODO: update test to be robust to station changes. Also, print whether the query was successful at all to just check package install

import os.path as osp
import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")

# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import str2time, hash_ndarray
import ingest.RAWS as rr

bbox = [42, -105, 45, -100]

# Stash Test: Expected Retrieved Stations (As of Oct 9, 2025)
stids = ['BRLW4', 'HRSN1', 'DOHS2', 'BKFS2', 'CRRS2', 'NMOS2', 'RDCS2', 'VRFN1', 'PINS2', 'DVLW4', 'WCAS2', 'RHRS2', 'TS485', 'WPKS2', 'CSPS2', 'SDSS2', 'RWES2', 'MTRN1', 'MKVN1', 'SFRS2', 'TT591']
fm_hash_expected = "ea28f3c6929ccc5574b8ed3587cb3759"

# API Test: Expected Retrieved Stations (As of Oct 9, 2025)
stids2 = ['BRLW4', 'HRSN1', 'DOHS2', 'BKFS2', 'CRRS2', 'NMOS2', 'RDCS2', 'VRFN1', 'PINS2', 'DVLW4', 'TS399', 'AGTN1', 'WCAS2', 'RHRS2', 'TS485', 'WPKS2', 'CSPS2', 'SDSS2', 'RWES2', 'TT456', 'MTRN1', 'MKVN1', 'SFRS2']
fm_hash_expected2="3d249ee136c1c5633a2526d022e6e982"


if __name__ == '__main__':

    print(f"Testing RAWS Retrieval from Stash")
    start = str2time('2024-01-01T00:00:00Z')
    end = str2time('2024-01-01T02:00:00Z')
    raws_dict = rr.build_raws_dict_stash(start, end, bbox)
    fm = pd.concat([raws_dict[st]["RAWS"]["fm"] for st in raws_dict]).reset_index(drop=True).astype(float).to_numpy().round(3)
    fm_hash = hash_ndarray(fm)
    
    failed=False
    if [*raws_dict.keys()] != stids:
        warnings.warn("Returned STIDs don't match expected")
        failed=True
    if fm_hash != fm_hash_expected:
        warnings.warn("FM hash doesn't match expected")
        failed=True
    if not failed:
        print(f"TEST PASSED STASH: stashed RAWS dictionary matches expected stations, FM data hash matches") 
    
    print()
    print(f"Testing RAWS Retrieval from API")
    start = str2time('2024-12-30T00:00:00Z')
    end = str2time('2024-12-30T02:00:00Z')
    
    oneyr = datetime.now(timezone.utc) - timedelta(days=365)
    if start <= oneyr:
        warnings.warn("Start date not within the past year, free Synoptic API will fail")

    raws_dict = rr.build_raws_dict_api(start, end, bbox)
    fm = pd.concat([raws_dict[st]["RAWS"]["fm"] for st in raws_dict]).reset_index(drop=True).astype(float).to_numpy().round(3)    
    fm_hash = hash_ndarray(fm)

    failed=False
    if [*raws_dict.keys()] != stids2:
        warnings.warn("Returned STIDs don't match expected")
        failed=True
    if fm_hash != fm_hash_expected2:
        warnings.warn("FM hash doesn't match expected")
        failed=True
    if not failed:
        print(f"TESTS PASSED API: RAWS dictionary matches expected stations, FM data hash matches")    



