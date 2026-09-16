# src/data/ohe.py
import polars as pl
from src.config.categories import OHE_CATEGORIES

def one_hot_encode(df: pl.DataFrame | pl.LazyFrame,):
    expressions = []

    for feature, categories in OHE_CATEGORIES.items():

        for category in categories:

            encoded_name = f"{feature}_{category}"

            expressions.append(
                (
                    pl.col(feature).cast(pl.String)
                    == pl.lit(category)
                )
                .fill_null(False)
                .cast(pl.UInt8)
                .alias(encoded_name)
            )

    return (df
            .with_columns(expressions)
            .select(pl.exclude(OHE_CATEGORIES.keys())))