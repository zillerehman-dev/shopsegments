"""Project paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "online_retail.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"

RANDOM_STATE = 42
RFM_FEATURES = ["Recency", "Frequency", "Monetary"]

# Hold-out split for the recommender: train on everything before this date,
# evaluate on what customers bought afterwards (last ~3 months of data).
RECO_SPLIT_DATE = "2011-09-09"
TOP_K = 10
