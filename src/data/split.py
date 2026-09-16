# src/data/split.py
import polars as pl
from src.config.features import PERIOD_COL, TARGET_ENG


def temporal_split(
    df: pl.LazyFrame,
    test_start: str = "2019-04-01",
    val_start: str  = "2018-08-01",
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Temporal train/val/test split — no shuffling, respects time order.

    train : issue_d <  val_start
    val   : val_start <= issue_d < test_start
    test  : issue_d >= test_start
    """
    val_date  = pl.date(*[int(x) for x in val_start.split("-")])
    test_date = pl.date(*[int(x) for x in test_start.split("-")])

    df_train = df.filter(pl.col(PERIOD_COL) < val_date)
    df_val   = df.filter(
        (pl.col(PERIOD_COL) >= val_date) &
        (pl.col(PERIOD_COL) < test_date)
    )
    df_test  = df.filter(pl.col(PERIOD_COL) >= test_date)

    return df_train, df_val, df_test


def split_xy(
    df: pl.LazyFrame,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Split features and target."""
    X = df.drop([TARGET_ENG, PERIOD_COL])
    y = df.select([TARGET_ENG])
    return X, y