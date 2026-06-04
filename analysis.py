import pandas as pd


def load_data(file):
    return pd.read_csv(file)


def dataset_info(df):

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicates": df.duplicated().sum()
    }


def summary_statistics(df):
    return df.describe()


def missing_values(df):
    return df.isnull().sum()
