# src/features/scaling.py
import pickle
import polars as pl
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from src.config.features import (
    TARGET_ENG, PERIOD_COL,
    ENG_NULL_MTHS, ENG_CR_LINE,
    FINAL_FEATURES,
)
from src.config.paths import ARTIFACT_PREPROCESS_DIR

SCALER_PATH = ARTIFACT_PREPROCESS_DIR / "scaler.pkl"


def _get_scale_features(df: pl.LazyFrame) -> list[str]:
    """
    Derive scale features dynamically from the actual post-cleansing schema.
    Excludes:
      - target and period columns
      - binary null indicators (_is_null suffix or dtype UInt8/Int8 with only 0/1)
      - categorical columns (already woe-encoded at this point, 
        but guard in case any remain)
    """
    schema = df.collect_schema()

    numeric_types = {
        pl.Float64, pl.Float32,
        pl.Int64, pl.Int32, pl.Int16,
    }

    exclude = {TARGET_ENG, PERIOD_COL}

    # detect binary cols by suffix OR dtype UInt8
    binary_cols = {
        col for col, dtype in schema.items()
        if col.endswith("_is_null")
        or dtype == pl.UInt8
    }

    return [
        col for col, dtype in schema.items()
        if dtype in numeric_types
        and col not in exclude
        and col not in binary_cols
    ]


class FeatureScaler:
    """
    Fits StandardScaler on train post-cleansing and post-WoE schema.
    Derives scale features dynamically from the actual dataframe schema
    rather than the raw config schema, so it correctly handles:
      - woe-encoded columns (new float cols replacing categoricals)
      - engineered columns (cr_age_mths, _is_null indicators)
      - any schema changes introduced during cleansing
    """

    def __init__(self):
        self.scaler:   RobustScaler = RobustScaler()
        self.features: list[str]      = []
        self._fitted:  bool           = False

    def fit_transform(self, df: pl.LazyFrame) -> pl.LazyFrame:
        self.features = _get_scale_features(df)

        pdf = df.collect().to_pandas()
        pdf[self.features] = self.scaler.fit_transform(pdf[self.features])

        self._fitted = True
        self.save()

        return pl.from_pandas(pdf).lazy()

    def transform(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if not self._fitted:
            raise RuntimeError(
                "FeatureScaler is not fitted. "
                "call fit_transform on train first or load() a saved scaler."
            )

        # guard: check all expected features exist in incoming df
        schema_cols = set(df.collect_schema().names())
        missing = [f for f in self.features if f not in schema_cols]
        if missing:
            raise ValueError(f"columns missing from dataframe: {missing}")

        pdf = df.collect().to_pandas()
        pdf[self.features] = self.scaler.transform(pdf[self.features])

        return pl.from_pandas(pdf).lazy()

    def save(self, path: Path = SCALER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"scaler saved to {path}")

    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> "FeatureScaler":
        if not path.exists():
            raise FileNotFoundError(f"no scaler found at {path}")
        with open(path, "rb") as f:
            instance = pickle.load(f)
        print(f"scaler loaded from {path}")
        return instance

    def summary(self) -> pd.DataFrame:
        """Return fitted mean and std per feature for inspection."""
        if not self._fitted:
            raise RuntimeError("scaler not fitted yet.")
        return pd.DataFrame({
            "feature": self.features,
            "mean":    self.scaler.mean_,
            "std":     self.scaler.scale_,
        }).sort_values("std", ascending=False).reset_index(drop=True)