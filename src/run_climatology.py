import sys
import pickle
import os.path as osp


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
from utils import parse_bbox
from models.moisture_models import build_climatology, get_climatology_forecasts

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print(f"Invalid arguments. {len(sys.argv)} was given but 4 expected")
        print(('Usage: %s <esmf_from_utc> <esmf_to_utc> <bbox> <output_file>' % sys.argv[0]))
        print("Example: python src/ingest/build_fmda_dicts.py '2023-06-01T00:00:00Z' '2023-06-01T05:00:00Z' '[37,-105,39,-103]' test_climatology.pkl")
        print("bbox format should match rtma_cycler: [latmin, lonmin, latmax, lonmax]")
        sys.exit(-1)

    start = sys.argv[1]
    end = sys.argv[2]
    bbox = parse_bbox(sys.argv[3])
    out_file = sys.argv[4]

    clim_dict = build_climatology(
        start,
        end,
        bbox
    )
    
    with open(out_file, 'wb') as handle:
        pickle.dump(clim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    