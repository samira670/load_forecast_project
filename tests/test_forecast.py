import pandas as pd
from src.train_forecast import create_feats

def test_create_feats():
    ts = pd.date_range("2023-01-01", periods=200, freq="H")
    df = pd.DataFrame({"timestamp": ts, "load_MW": range(200)})
    feats = create_feats(df)
    assert "lag_1" in feats.columns
    assert not feats.isnull().any().any()
