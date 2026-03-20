"""Basic settings used by most runs."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"

INSTANCE_DIRS = [
    PROJECT_ROOT / "data" / "instances-set1",
    PROJECT_ROOT / "data" / "instances-set2",
]

INSTANCE_NAMES = list("ABCDEFGHIJK")

BEST_KNOWN_SOFT = {
    "A": 515,
    "B": 691,
    "C": 699,
    "D": 944,
    "E": 375,
    "F": 704,
}

BEST_KNOWN_HARD = {
    "A": 522,
    "B": 714,
    "C": 724,
    "D": 972,
    "E": 375,
    "F": 721,
}