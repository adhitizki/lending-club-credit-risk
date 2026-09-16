# src/features/woe.py
import pickle
import warnings
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from src.config.features import TARGET_ENG, PERIOD_COL
from src.config.paths import ARTIFACT_PREPROCESS_DIR

WOE_PATH = ARTIFACT_PREPROCESS_DIR / "woe_encoder.pkl"


def _get_woe_features(df: pl.LazyFrame) -> list[str]:
    """
    Derive categorical features dynamically from post-cleansing schema.
    Excludes target, period, and any already-numeric columns.
    """
    schema = df.collect_schema()

    categorical_types = (pl.Categorical, pl.String)
    exclude = {TARGET_ENG, PERIOD_COL}

    return [
        col for col, dtype in schema.items()
        if dtype in categorical_types
        and col not in exclude
    ]


def _iv_label(iv: float) -> str:
    if iv < 0.02:   return "useless"
    if iv < 0.1:    return "weak"
    if iv < 0.3:    return "medium"
    if iv < 0.5:    return "strong"
    return "suspicious"


class WoEEncoder:
    """
    Weight of Evidence encoder. Fits on train only, applies to val/test.

    Derives categorical features dynamically from the actual post-cleansing
    schema rather than raw config, so it correctly handles:
      - any categoricals surviving cleansing
      - new categorical columns introduced during engineering
      - columns dropped during cleansing that no longer exist

    WoE  = ln(Distribution of Events / Distribution of Non-Events)
    IV   = sum((p_event - p_non_event) * WoE) per feature

    IV interpretation:
      < 0.02  useless
      < 0.10  weak
      < 0.30  medium
      < 0.50  strong
      >= 0.50 suspicious (may indicate leakage)
    """

    def __init__(self, min_samples: int = 50):
        self.min_samples:  int               = min_samples
        self.features:     list[str]         = []
        self.woe_maps_:    dict[str, dict]   = {}
        self.iv_:          dict[str, float]  = {}
        self._fitted:      bool              = False

    def fit(self, df: pl.LazyFrame) -> "WoEEncoder":

        self.features = _get_woe_features(df)

        if not self.features:
            warnings.warn("no categorical features found to encode.")
            self._fitted = True
            self.save()
            return self

        pdf = (
            df
            .select(self.features + [TARGET_ENG])
            .collect()
            .to_pandas()
        )

        total_events     = max(pdf[TARGET_ENG].sum(), 1)
        total_non_events = max(len(pdf) - total_events, 1)

        for col in self.features:
            stats = (
                pdf.groupby(col, dropna=False)[TARGET_ENG]
                .agg(["sum", "count"])
                .rename(columns={"sum": "events", "count": "total"})
                .reset_index()
            )

            # rename NaN category to explicit missing label
            stats[col] = stats[col].fillna("__missing__")

            stats["non_events"]   = stats["total"] - stats["events"]
            stats["p_event"]      = (stats["events"] / total_events).clip(lower=1e-10)
            stats["p_non_event"]  = (stats["non_events"] / total_non_events).clip(lower=1e-10)
            stats["woe"]          = np.log(stats["p_event"] / stats["p_non_event"])
            stats["iv_component"] = (stats["p_event"] - stats["p_non_event"]) * stats["woe"]

            # zero out bins with too few samples — unreliable woe estimate
            stats.loc[stats["total"] < self.min_samples, "woe"] = 0.0
            stats.loc[stats["total"] < self.min_samples, "iv_component"] = 0.0

            self.woe_maps_[col] = dict(zip(stats[col], stats["woe"]))
            self.iv_[col]       = stats["iv_component"].sum()

        self._fitted = True
        self.save()
        return self

    def transform(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if not self._fitted:
            raise RuntimeError(
                "WoEEncoder is not fitted. "
                "call fit on train first or load() a saved encoder."
            )

        if not self.features:
            return df

        # guard: check all expected features exist
        schema_cols = set(df.collect_schema().names())
        missing_cols = [f for f in self.features if f not in schema_cols]
        if missing_cols:
            raise ValueError(f"columns missing from dataframe: {missing_cols}")

        pdf = df.collect().to_pandas()

        for col in self.features:
            # fill nulls with missing label before mapping
            pdf[col] = pdf[col].fillna("__missing__")

            new_col     = f"{col}_woe"
            # unseen categories in val/test get woe 0.0
            pdf[new_col] = (
                pdf[col]
                .map(self.woe_maps_[col])
                .fillna(0.0)
            )

        # drop original categorical columns
        pdf = pdf.drop(columns=self.features)
        return pl.from_pandas(pdf).lazy()

    def fit_transform(self, df: pl.LazyFrame) -> pl.LazyFrame:
        return self.fit(df).transform(df)

    def iv_summary(self) -> pd.DataFrame:
        """
        Return IV table sorted by predictive power.
        Flag suspicious IV (>=0.5) as potential leakage.
        """
        if not self._fitted:
            raise RuntimeError("encoder not fitted yet.")

        df = (
            pd.DataFrame
            .from_dict(self.iv_, orient="index", columns=["iv"])
            .reset_index()
            .rename(columns={"index": "feature"})
            .sort_values("iv", ascending=False)
            .reset_index(drop=True)
        )
        df["predictive_power"] = df["iv"].apply(_iv_label)

        # flag suspicious
        suspicious = df[df["predictive_power"] == "suspicious"]["feature"].tolist()
        if suspicious:
            warnings.warn(
                f"high IV features (>=0.5) detected, possible leakage: {suspicious}"
            )

        return df

    def woe_summary(self, feature: str) -> pd.DataFrame:
        """
        Return per-bin WoE table for a single feature.
        Useful for binning inspection and scorecard development.
        """
        if not self._fitted:
            raise RuntimeError("encoder not fitted yet.")
        if feature not in self.woe_maps_:
            raise ValueError(f"{feature} not in fitted features: {list(self.woe_maps_.keys())}")

        return (
            pd.DataFrame
            .from_dict(self.woe_maps_[feature], orient="index", columns=["woe"])
            .reset_index()
            .rename(columns={"index": "category"})
            .sort_values("woe", ascending=False)
            .reset_index(drop=True)
        )

    def drop_weak_features(self, threshold: float = 0.02) -> list[str]:
        """
        Return list of features with IV below threshold.
        Use before modelling to drop useless features.
        """
        if not self._fitted:
            raise RuntimeError("encoder not fitted yet.")
        return [
            f"{col}_woe" for col, iv in self.iv_.items()
            if iv < threshold
        ]

    def save(self, path: Path = WOE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"woe encoder saved to {path}")

    @classmethod
    def load(cls, path: Path = WOE_PATH) -> "WoEEncoder":
        if not path.exists():
            raise FileNotFoundError(f"no woe encoder found at {path}")
        with open(path, "rb") as f:
            instance = pickle.load(f)
        print(f"woe encoder loaded from {path}")
        return instance