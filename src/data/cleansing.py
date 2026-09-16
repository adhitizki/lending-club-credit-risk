# src/data/cleansing.py
import polars as pl
from src.config.features import (
    KEEP_FEATURES,
    PERIOD_COL,
    CUTOFF_DATE,
    TARGET,
    TARGET_ENG,
    TARGET_MAP,
    ENG_FILL_MTHS,
    ENG_NULL_MTHS,
    ENG_DROP,
    ENG_CR_LINE,
    IMPUTER_SKIP,
    FINAL_FEATURES,
    SENTINEL_MTHS,
)

def _to_lazy(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return df.lazy() if isinstance(df, pl.DataFrame) else df

def feature_selection(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Select feature based on previous performance.
    """
    df = _to_lazy(df)
    return df.select(KEEP_FEATURES + [PERIOD_COL, TARGET])

def filter_sample(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Filter to resolved loans from Jan 2016 onwards.
    - removes unresolved (current, in grace period, late) loans
    - removes pre-2016 data where recording was unstable
    """
    df = _to_lazy(df)
    return (df
            .with_columns(
                pl.col(PERIOD_COL).str
                .strptime(pl.Date, "%b-%Y", strict=False)
                .alias(PERIOD_COL))
            .filter(
                (pl.col(PERIOD_COL) >= pl.date(*CUTOFF_DATE)) &
                (pl.col(TARGET).is_in(list(TARGET_MAP.keys())))
    ))

def normalization_casting(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Strip unnecessary character to cast into numeric features
    """
    df = _to_lazy(df)
    df = df.with_columns([

            # cast term to int
            pl.col("term")
            .str.strip_chars()
            .str.replace(" months", "")
            .cast(pl.Int8, strict=False),

            # cast int_rate
            pl.col("int_rate")
            .str.strip_chars()
            .str.replace("%", "")
            .cast(pl.Float64, strict=False),

            # cast revol_util
            pl.col("revol_util")
            .str.strip_chars()
            .str.replace("%", "")
            .cast(pl.Float64, strict=False),

            # cast emp_length
            pl.col("emp_length")
            .str.strip_chars()
            .str.replace(" years", "")
            .str.replace(" year", "")
            .str.replace("< 1", "0")
            .str.replace("10+", "10")
            .cast(pl.Int8, strict=False)
        ])

    return df

def engineer_target(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Consolidate loan_status into binary default flag.
    0 = paid, 1 = default
    """
    df = _to_lazy(df)
    return df.with_columns(
        pl.col(TARGET)
        .replace(TARGET_MAP)
        .cast(pl.Int8)
        .alias(TARGET_ENG)
    )


def engineer_cr_age(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Derive cr_age_mths from earliest_cr_line and issue_d.
    Measures how long borrower has had credit at time of application.
    """
    df = _to_lazy(df)
    return (df
            .with_columns(
                pl.col("earliest_cr_line")
                .str.strptime(pl.Date, "%b-%Y", strict=False))
            .with_columns(
            (
                (pl.col("issue_d") - pl.col("earliest_cr_line"))
                .dt.total_days() // 30
            )
            .cast(pl.Int64)
            .alias(ENG_CR_LINE)
    ))


def add_null_indicators(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Add binary null indicators for sparse mths_since columns before imputation.
    Preserves the signal that an event never occurred vs data missing.
    """
    df = _to_lazy(df)
    return df.with_columns([
        pl.col(col).is_null().cast(pl.Int8).alias(f"{col}_null")
        for col in ENG_NULL_MTHS
    ])


def fill_mths_sentinel(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Fill mths_since columns with sentinel value (999).
    Null here means the event never occurred, not missing data.
    """
    df = _to_lazy(df)
    return df.with_columns([
        pl.col(col).fill_null(SENTINEL_MTHS)
        for col in ENG_FILL_MTHS
    ])


def impute_by_issue_date(
    df: pl.DataFrame | pl.LazyFrame,
    special_features: set[str] | None = None,
) -> pl.LazyFrame:
    """
    Impute remaining nulls using per-issue_d statistics.
    Numeric: median by issue_d, fallback 0.
    Categorical: mode by issue_d, fallback 'Missing'.
    Skips special_features entirely.
    """
    df = _to_lazy(df)
    special_features = special_features or set()
    schema = df.collect_schema()

    numeric_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    categorical_types = {pl.String, pl.Categorical}

    null_counts = (
        df
        .select([pl.col(c).null_count().alias(c) for c in schema.names()])
        .collect()
        .row(0, named=True)
    )

    numeric_features = [
        c for c, cnt in null_counts.items()
        if cnt > 0
        and c != PERIOD_COL
        and c not in special_features
        and schema[c] in numeric_types
    ]

    categorical_features = [
        c for c, cnt in null_counts.items()
        if cnt > 0
        and c != PERIOD_COL
        and c not in special_features
        and schema[c] in categorical_types
    ]

    if numeric_features:
        numeric_stats = (
            df.group_by(PERIOD_COL)
            .agg([
                pl.col(f).median().alias(f"_impute_{f}")
                for f in numeric_features
            ])
        )
        df = (
            df
            .join(numeric_stats, on=PERIOD_COL, how="left")
            .with_columns([
                pl.coalesce(
                    pl.col(f),
                    pl.col(f"_impute_{f}"),
                    pl.lit(0).cast(schema[f]),
                ).alias(f)
                for f in numeric_features
            ])
            .drop([f"_impute_{f}" for f in numeric_features])
        )

    if categorical_features:
        categorical_stats = (
            df.group_by(PERIOD_COL)
            .agg([
                pl.col(f).drop_nulls().mode().first().alias(f"_impute_{f}")
                for f in categorical_features
            ])
        )
        df = (
            df
            .join(categorical_stats, on=PERIOD_COL, how="left")
            .with_columns([
                pl.coalesce(
                    pl.col(f),
                    pl.col(f"_impute_{f}"),
                    pl.lit("Missing"),
                ).alias(f)
                for f in categorical_features
            ])
            .drop([f"_impute_{f}" for f in categorical_features])
        )

    return df


def select_final(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Drop engineered-away columns and select final feature set.
    """
    df = _to_lazy(df)
    return (
        df
        .select(FINAL_FEATURES + [TARGET_ENG, PERIOD_COL])
    )


def cleanse(df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Full cleansing pipeline. Called by pipeline.py.
    Steps:
        1. filter sample (resolved loans, Jan 2016 onwards)
        2. engineer target (binary default flag)
        3. engineer cr_age_mths
        4. add null indicators for sparse mths_since columns
        5. fill mths_since with sentinel
        6. impute remaining nulls by issue_d
        7. select final feature set
    """
    return (
        _to_lazy(df)
        .pipe(feature_selection)
        .pipe(filter_sample)
        .pipe(engineer_target)
        .pipe(normalization_casting)
        .pipe(engineer_cr_age)
        .pipe(add_null_indicators)
        .pipe(fill_mths_sentinel)
        .pipe(impute_by_issue_date, special_features=IMPUTER_SKIP)
        .pipe(select_final)
    )