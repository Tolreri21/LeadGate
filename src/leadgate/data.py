import pandas as pd
from pathlib import Path


def load_raw(path):
    df = pd.read_csv(path, sep=";")
    return df


def clean_raw(df):
    df = df[~((df["poutcome"] == "unknown") & (df["pdays"] != -1))]
    df.drop(["pdays", "day", "duration"], axis=1, inplace=True)
    df["y"] = (df["y"] == "yes").astype(int)
    return df


def load_preprocessed(data_dir):
    p = Path(data_dir)
    X_train = pd.read_csv(p / "X_train.csv")
    X_test = pd.read_csv(p / "X_test.csv")
    y_train = pd.read_csv(p / "y_train.csv").squeeze()
    y_test = pd.read_csv(p / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test
