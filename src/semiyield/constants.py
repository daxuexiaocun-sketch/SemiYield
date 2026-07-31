from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
DATA_FILE_URLS = {
    "secom.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
    "secom_labels.data": (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    ),
    "secom.names": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.names",
}
DATASET_DOI = "10.24432/C54305"
SCHEMA_VERSION = "secom-uci-v1"
FEATURE_PREFIX = "sensor_"
DEFAULT_DATA_DIR = Path("data") / "raw"
DEFAULT_MODEL_PATH = Path("artifacts") / "model.joblib"
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42
NASA_MOSFET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/13.+MOSFET+Thermal+Overstress+Aging.zip"
)
NASA_DATASET_CITATION = (
    "J. R. Celaya, A. Saxena, S. Saha, and K. Goebel, "
    "MOSFET Thermal Overstress Aging Data Set, NASA Ames Research Center."
)
EXPERIMENT_SCHEMA_VERSION = "semiyield-experiment-v1"
MOSFET_FEATURE_SCHEMA_VERSION = "nasa-mosfet-derived-v1"
