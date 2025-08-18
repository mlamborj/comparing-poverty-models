from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"

# data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

# external datasets
COUNTRIES_FILE = EXTERNAL_DIR / "country_list.csv"
SMOD_FILE = EXTERNAL_DIR / "SMOD_Africa.tif"
GADM_FILE = EXTERNAL_DIR / "africa_gadm36.gpkg"

# model lists
MODEL_NAMES = ["Chi", "Lee", "McCallum", "Yeh"]
MODEL_PAIRS = [
    "Chi_Yeh",
    "Lee_Chi",
    "Lee_Yeh",
    "Lee_McCallum",
    "McCallum_Chi",
    "McCallum_Yeh",
]
