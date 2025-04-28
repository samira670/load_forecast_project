"""
train_forecast.py  –  Quantile LightGBM models + anomaly flags
writes:  data/clean/preds.parquet
Samira · 2025
"""
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import IsolationForest

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
LONG_PARQ = CLEAN / "load_long.parquet"
OUT_PARQ  = CLEAN / "preds.parquet"

print("🔄 loading long parquet …")
df = pd.read_parquet(LONG_PARQ)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ─────────────────────────────────────────────────────────────
# Keep only **real** region names (letters / space)
# ─────────────────────────────────────────────────────────────
valid_regions = [r for r in df.region.unique()
                 if r.replace(" ", "").isalpha()]
print("✅ Regions to model:", ", ".join(valid_regions))

def create_feats(group):
    """Return feature DataFrame with target y column."""
    s = group.set_index("timestamp")["load_MW"].copy()
    feat = pd.DataFrame(index=s.index)
    feat["hour"]   = feat.index.hour
    feat["dow"]    = feat.index.dayofweek
    feat["month"]  = feat.index.month
    for lag in (1, 24, 168):
        feat[f"lag_{lag}"] = s.shift(lag)
    feat["roll_24"] = s.rolling(24).mean()
    feat["y"] = s
    return feat.dropna()

records = []

for region in valid_regions:
    sub = df[df.region == region]
    feats = create_feats(sub)

    if len(feats) < 500:
        print(f"⚠️ skipping {region} (only {len(feats)} rows)")
        continue

    X = feats.drop(columns="y")
    y = feats["y"]

    split = int(len(X) * 0.7)            # 70 % train, 30 % test
    X_tr, y_tr = X.iloc[:split], y.iloc[:split]
    X_te, y_te = X.iloc[split:], y.iloc[split:]

    def fit_q(alpha):
        dtrain = lgb.Dataset(X_tr, y_tr)
        return lgb.train(
            params=dict(objective="quantile",
                        alpha=alpha,
                        learning_rate=0.05,
                        num_leaves=64,
                        metric="quantile",
                        verbose=-1),
            train_set=dtrain,
            num_boost_round=400
        )

    print(f"🛠  training {region:15} …")
    mdl10, mdl50, mdl90 = (fit_q(a) for a in (0.1, 0.5, 0.9))
    p10, p50, p90 = (m.predict(X_te) for m in (mdl10, mdl50, mdl90))

    # Anomaly on residual
    resid = y_te - p50
    iso   = IsolationForest(contamination=0.01, random_state=42)
    flags = iso.fit_predict(resid.values.reshape(-1, 1)) == -1

    records.append(pd.DataFrame({
        "timestamp": X_te.index,
        "region"   : region,
        "actual"   : y_te.values,
        "p10"      : p10,
        "p50"      : p50,
        "p90"      : p90,
        "anomaly"  : flags,
    }))

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
preds = pd.concat(records, ignore_index=True)
preds.to_parquet(OUT_PARQ, index=False)
print(f"🎉 wrote {OUT_PARQ}   ({len(preds):,} rows  ·  {preds.region.nunique()} regions)")
