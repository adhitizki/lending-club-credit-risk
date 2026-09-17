# src/data/pipeline.py
from src.data.loader import load_raw
from src.data.cleansing import cleanse
from src.data.split import temporal_split, split_xy
from src.data.ohe import one_hot_encode
from src.data.woe import WoEEncoder
from src.data.scaling import FeatureScaler
from src.config.paths import PROCESSED_DIR
from src.param.data_pipeline import VAL_START, TEST_START, ENCODING, SCALING
from src.utils.logger import step, log_shape, shape, fmt_time, header

from typing import Literal
import time
import polars as pl


def run(
    val_start:  str                    = VAL_START,
    test_start: str                    = TEST_START,
    encoding:   Literal["woe", "ohe"] = ENCODING,
    scaling:    bool                   = SCALING,
) -> dict:

    pipeline_start = time.time()
    artifacts      = {}
    WIDTH          = 60

    # ── pipeline header ───────────────────────────────────────────────────────
    header("PIPELINE START", width=WIDTH)
    print(f"  encoding   : {encoding.upper()}")
    print(f"  scaling    : {scaling}")
    print(f"  val_start  : {val_start}")
    print(f"  test_start : {test_start}")

    # ── 1. load ───────────────────────────────────────────────────────────────
    with step(1, "LOAD RAW DATA", width=WIDTH):
        df = load_raw()
        log_shape("raw", df)

    # ── 2. cleanse ────────────────────────────────────────────────────────────
    with step(2, "CLEANSING", width=WIDTH):
        df = cleanse(df)
        log_shape("cleansed", df)

    # ── 3. temporal split ─────────────────────────────────────────────────────
    with step(3, "TEMPORAL SPLIT", width=WIDTH):
        df_train, df_val, df_test = temporal_split(
            df,
            test_start=test_start,
            val_start=val_start,
        )
        log_shape("train", df_train)
        log_shape("val",   df_val)
        log_shape("test",  df_test)

    # ── 4. encoding ───────────────────────────────────────────────────────────
    with step(4, f"ENCODING ({encoding.upper()})", width=WIDTH):

        if encoding == "ohe":
            df_train = one_hot_encode(df_train)
            df_val   = one_hot_encode(df_val)
            df_test  = one_hot_encode(df_test)

        elif encoding == "woe":
            woe      = WoEEncoder(min_samples=2)
            df_train = woe.fit_transform(df_train)
            df_val   = woe.transform(df_val)
            df_test  = woe.transform(df_test)

            print(f"\n    IV Summary:")
            print(woe.iv_summary().to_string(index=False))

            weak_cols = woe.drop_weak_features(threshold=0.02)
            if weak_cols:
                print(f"\n    dropping {len(weak_cols)} weak IV features: {weak_cols}")
                df_train = df_train.drop(weak_cols)
                df_val   = df_val.drop(weak_cols)
                df_test  = df_test.drop(weak_cols)
            else:
                print("\n    no weak features to drop.")

            artifacts["woe"] = woe

        else:
            raise ValueError(
                f"unknown encoding: '{encoding}'. expected 'woe' or 'ohe'."
            )

        print()
        log_shape("train encoded", df_train)
        log_shape("val encoded",   df_val)
        log_shape("test encoded",  df_test)

    # ── 5. split X y ─────────────────────────────────────────────────────────
    with step(5, "SPLIT X / Y", width=WIDTH):
        X_train, y_train = split_xy(df_train)
        X_val,   y_val   = split_xy(df_val)
        X_test,  y_test  = split_xy(df_test)

        print("    features:")
        log_shape("X_train", X_train)
        log_shape("X_val",   X_val)
        log_shape("X_test",  X_test)
        print("    target:")
        log_shape("y_train", y_train)
        log_shape("y_val",   y_val)
        log_shape("y_test",  y_test)

    # ── 6. scaling ────────────────────────────────────────────────────────────
    with step(6, f"SCALING (enabled: {scaling})", width=WIDTH):
        if scaling:
            scaler  = FeatureScaler()
            X_train = scaler.fit_transform(X_train)
            X_val   = scaler.transform(X_val)
            X_test  = scaler.transform(X_test)

            log_shape("X_train scaled", X_train)
            log_shape("X_val scaled",   X_val)
            log_shape("X_test scaled",  X_test)

            artifacts["scaler"] = scaler
        else:
            print("    scaling skipped.")

    # ── 7. save splits ────────────────────────────────────────────────────────
    splits = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
    }

    with step(7, "SAVE SPLITS", width=WIDTH):
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for name, frame in splits.items():
            path       = PROCESSED_DIR / f"{name}.parquet"
            save_start = time.time()
            frame.collect().write_parquet(path)
            rows, cols = shape(frame)
            elapsed    = fmt_time(time.time() - save_start)
            print(
                f"    ✓ {name:<14}"
                f"  rows: {rows:>10,}"
                f"  cols: {cols:>4}"
                f"  time: {elapsed:<8}"
                f"  -> {path.name}"
            )

    # ── pipeline summary ──────────────────────────────────────────────────────
    total = fmt_time(time.time() - pipeline_start)
    header(f"PIPELINE COMPLETE   total time: {total}", width=WIDTH)

    return {**splits, **artifacts}


if __name__ == "__main__":
    run()