"""Read a saved csv and print it again for quick checking."""

from pathlib import Path
import pandas as pd

from configs.default import TABLES_DIR

if __name__ == "__main__":
    path = TABLES_DIR / "soft_cluvrp_sa_exact_results_A_to_K.csv"
    df = pd.read_csv(path)
    print(df)