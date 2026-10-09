from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """
    Učitava dataset iz CSV fajla.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset ne postoji: {path}")

    return pd.read_csv(path)


def inspect_data(df: pd.DataFrame) -> None:
    """
    Ispisuje osnovne informacije o datasetu.
    """
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nTarget distribution:")
    print(df["Revenue"].value_counts(dropna=False))

    print("\nTarget distribution in percentages:")
    print(df["Revenue"].value_counts(normalize=True, dropna=False) * 100)

    print("\nFirst rows:")
    print(df.head())

    print("\nNumerical statistics:")
    print(df.describe().T)