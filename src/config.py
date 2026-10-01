"""Project-wide paths and constants.

Everything that a reader might want to tweak (folders, class codes, default
parameters) lives here so it is not scattered across the code base.
"""

from pathlib import Path

# --- Folders -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_TILES_DIR = DATA_DIR / "raw_tiles"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "lidar_classifier.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "lidar_classifier_metrics.json"

# --- Classes -----------------------------------------------------------------

# Our 4 target classes (roadmap, phase 2).
GROUND, VEGETATION, BUILDING, OTHER = 0, 1, 2, 3
CLASS_NAMES = {GROUND: "ground", VEGETATION: "vegetation", BUILDING: "building", OTHER: "other"}

# ASPRS codes found in the Montreal tiles (see docs/01_data.md).
ASPRS_UNCLASSIFIED = 1
ASPRS_GROUND = 2
ASPRS_BUILDING = 6
ASPRS_NOISE = 7
ASPRS_BRIDGE_DECK = 17

# Unclassified points higher than this above the ground are labelled as
# vegetation when we build training labels (the vendor did not classify trees).
VEGETATION_MIN_HEIGHT_M = 2.0

# --- Processing defaults -----------------------------------------------------

DTM_RESOLUTION_M = 1.0          # cell size of the ground model used for normalisation
DENSITY_RADIUS_M = 1.0          # radius of the sphere used for "local density"
NEIGHBOURS_K = 16               # neighbours used for the geometric features

BUILDING_GRID_M = 1.0           # raster cell size used to segment buildings
BUILDING_MIN_AREA_M2 = 25.0     # smaller blobs are ignored (cars, sheds, noise)
BUILDING_HEIGHT_PERCENTILE = 98 # robust "roof height" (ignores antennas / outliers)
BUILDING_MIN_PURITY = 0.4       # min share of building points among points > 2 m in the footprint

# Largest area a single request may process, to keep runtimes reasonable
# on a laptop (one full tile is 1 km²).
MAX_AREA_KM2 = 1.0

# --- Validation (phase 6) ----------------------------------------------------

# The tallest building in these tiles is well under 100 m, and the tallest in
# Montreal is ~205 m (1000 De La Gauchetière). 150 m follows the roadmap; raise
# it before analysing downtown.
MAX_BUILDING_HEIGHT_M = 150.0
MAX_VALIDATION_RETRIES = 2
