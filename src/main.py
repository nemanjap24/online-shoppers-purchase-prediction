from data_preparation import load_data, inspect_data


def main():
    df = load_data()
    inspect_data(df)


if __name__ == "__main__":
    main()