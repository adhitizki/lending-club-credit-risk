# src/data/loader.py
import polars as pl
from src.config.paths import RAW_DATA_FILE

def load_raw() -> pl.LazyFrame:

    return pl.scan_csv(RAW_DATA_FILE,
                       infer_schema_length=10000,
                       ignore_errors=True)