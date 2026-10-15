from data_preparation import (
    BOUNDED_COLUMNS,
    DURATION_COLUMNS,
    PAGE_COUNT_COLUMNS,
    clean_data,
    inspect_data,
    load_data,
)


def main():
    df_original = load_data()

    print("ORIGINAL DATASET")
    inspect_data(df_original)

    df_clean = clean_data(df_original)

    print("\n\nCLEANED DATASET")
    inspect_data(df_clean)

    assert "SessionID" not in df_clean.columns
    assert df_clean.isna().sum().sum() == 0

    for column in DURATION_COLUMNS + PAGE_COUNT_COLUMNS:
        assert (df_clean[column] >= 0).all()

    for column in BOUNDED_COLUMNS:
        assert ((df_clean[column] >= 0) & (df_clean[column] <= 1)).all()

    print("\nData validation passed.")


if __name__ == "__main__":
    main()