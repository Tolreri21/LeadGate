from leadgate.data import clean_raw, load_preprocessed
import pandas as pd


def test_cleaning(raw_df):
    df = raw_df
    out = clean_raw(df)
    assert not {"pdays", "day", "duration"} & set(out.columns)
    assert out["y"].dtype == int and set(out["y"].unique()) == {1, 0}
    assert len(out) == 3
    assert (out["balance"] < 0).any()


def test_load_preprocessed(tmp_path):
    # Кладём минимальный сплит в tmp_path, как его пишут ноутбуки (to_csv, index=False).
    X = pd.DataFrame({"age": [30, 40], "job": ["admin.", "retired"]})
    y = pd.Series([1, 0], name="y")
    X.to_csv(tmp_path / "X_train.csv", index=False)
    X.to_csv(tmp_path / "X_test.csv", index=False)
    y.to_csv(tmp_path / "y_train.csv", index=False)
    y.to_csv(tmp_path / "y_test.csv", index=False)

    X_train, X_test, y_train, y_test = load_preprocessed(tmp_path)

    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)
    assert len(X_train) == 2
