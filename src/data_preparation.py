from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "online_shoppers.csv"


DURATION_COLUMNS = [
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
]

PAGE_COUNT_COLUMNS = [
    "Administrative",
    "Informational",
    "ProductRelated",
]

RATE_COLUMNS = [
    "BounceRates",
    "ExitRates",
]

BOUNDED_COLUMNS = [
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]


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


def find_invalid_rows(df: pd.DataFrame) -> pd.Series:
    """
    Pronalazi redove sa vrednostima koje krše semantička ograničenja
    definisana opisom atributa.
    """
    invalid_mask = pd.Series(False, index=df.index)

    for column in DURATION_COLUMNS + PAGE_COUNT_COLUMNS:
        invalid_mask |= df[column] < 0

    for column in BOUNDED_COLUMNS:
        invalid_mask |= (df[column] < 0) | (df[column] > 1)

    return invalid_mask


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Čisti dataset:
    - uklanja SessionID,
    - uklanja redove sa nevalidnim vrednostima,
    - popunjava missing vrednosti.
    """
    cleaned = df.copy()

    invalid_mask = find_invalid_rows(cleaned)
    invalid_count = int(invalid_mask.sum())

    print(f"Invalid rows removed: {invalid_count}")

    cleaned = cleaned.loc[~invalid_mask].copy()

    if "SessionID" in cleaned.columns:
        cleaned = cleaned.drop(columns=["SessionID"])

    cleaned["Informational_Duration"] = (
        cleaned["Informational_Duration"].fillna(0)
    )

    cleaned["ProductRelated_Duration"] = (
        cleaned["ProductRelated_Duration"].fillna(0)
    )

    cleaned["VisitorType"] = (
        cleaned["VisitorType"].fillna("Unknown")
    )

    return cleaned.reset_index(drop=True)