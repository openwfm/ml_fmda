# Code to handle joining landfire data to HRRR and RAWS

import xarray as xr
from herbie import Herbie
import numpy as np
import sys
sys.path.append("src")
from utils import str2time

# Read LandFire
lf = xr.open_dataset("data/lf_elevation_hrrrgrid.tif")

# Get a Herbie file to line up grids and use pick_points method
# Get HRRR orographic height and land-sea mask
start = str2time('2024-12-31T00:00:00Z')
end = str2time('2024-12-31T02:00:00Z')
bbox = [40, -105, 45, -100]
H = Herbie("2025-01-01", product="prs")
hrrr = H.xarray("(?:HGT|LAND):surface")


# Add LF elevation as data variable
lf_elev = lf.band_data.squeeze(dim='band').transpose('y', 'x')
lf_elev = lf_elev.where(~np.isclose(lf_elev, -9999, atol=1e-3))
lf_elev = lf_elev.isel(y=slice(None, None, -1)) # flip y axis
lf_elev['y'] = lf_elev['y'][::-1] # adjust associated coordinates
hrrr = hrrr.assign(elev=lf_elev)
# Set to NA based on land-sea mask
# hrrr["orog"] = hrrr.orog.where(hrrr.lsm > 0)
# hrrr["elev"] = hrrr.elev.where(hrrr.lsm > 0)


