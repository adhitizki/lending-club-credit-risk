# config.py
import polars as pl

# ── period ────────────────────────────────────────────────────────────────────
PERIOD_COL    = "issue_d"
CUTOFF_DATE   = (2016, 1, 1)

# ── target ────────────────────────────────────────────────────────────────────
TARGET        = "loan_status"
TARGET_ENG    = "default"

TARGET_MAP = {
    "Fully Paid":                                         0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off":                                        1,
    "Default":                                            1,
    "Does not meet the credit policy. Status:Charged Off":1,
}

# ── sentinel for mths_since null fill ─────────────────────────────────────────
SENTINEL_MTHS = 0

# ── schema ────────────────────────────────────────────────────────────────────
SCHEMA: dict[str, pl.DataType] = {
    # loan info
    "loan_amnt":                   pl.Int64,
    "term":                        pl.Int64,
    "int_rate":                    pl.Float64,
    "grade":                       pl.Categorical,
    "initial_list_status":         pl.Categorical,
    "application_type":            pl.Categorical,
    "purpose":                     pl.Categorical,
    # borrower profile
    "emp_length":                  pl.Int64,
    "home_ownership":              pl.Categorical,
    "annual_inc":                  pl.Float64,
    "verification_status":         pl.Categorical,
    "addr_state":                  pl.Categorical,
    # credit score
    "fico_range_low":              pl.Int64,
    # credit history
    "earliest_cr_line":            pl.Date,
    "mo_sin_old_il_acct":          pl.Int64,
    "mo_sin_old_rev_tl_op":        pl.Int64,
    "mo_sin_rcnt_rev_tl_op":       pl.Int64,
    "mo_sin_rcnt_tl":              pl.Int64,
    "num_tl_op_past_12m":          pl.Int64,
    # credit profile
    "dti":                         pl.Float64,
    "total_acc":                   pl.Int64,
    "open_act_il":                 pl.Int64,
    "open_il_12m":                 pl.Int64,
    "open_rv_24m":                 pl.Int64,
    "all_util":                    pl.Float64,
    "total_cu_tl":                 pl.Int64,
    "pct_tl_nvr_dlq":              pl.Float64,
    "mort_acc":                    pl.Int64,
    "tot_cur_bal":                 pl.Float64,
    "num_actv_rev_tl":             pl.Int64,
    # revolving credit
    "revol_util":                  pl.Float64,
    "max_bal_bc":                  pl.Float64,
    "mths_since_recent_bc":        pl.Int64,
    "total_bc_limit":              pl.Float64,
    # installment credit
    "mths_since_rcnt_il":          pl.Int64,
    "total_bal_il":                pl.Float64,
    "il_util":                     pl.Float64,
    # credit inquiry
    "inq_last_6mths":              pl.Int64,
    "inq_fi":                      pl.Int64,
    "inq_last_12m":                pl.Int64,
    "mths_since_recent_inq":       pl.Int64,
    # delinquency
    "delinq_2yrs":                 pl.Int64,
    "mths_since_last_major_derog": pl.Int64,
    "num_accts_ever_120_pd":       pl.Int64,
    "num_tl_120dpd_2m":            pl.Int64,
    "num_tl_90g_dpd_24m":          pl.Int64,
    "collections_12_mths_ex_med":  pl.Int64,
    # public record
    "pub_rec":                     pl.Int64,
    "pub_rec_bankruptcies":        pl.Int64,
    "mths_since_last_record":      pl.Int64,
    # collection
    "tot_coll_amt":                pl.Float64,
    # installment credit (cont.)
    "num_il_tl":                   pl.Int64,
    # period and target
    "issue_d":                     pl.Date,
    "loan_status":                 pl.Categorical,
}

# ── feature groups ────────────────────────────────────────────────────────────

# features to keep from raw data (exclude period and target)
KEEP_FEATURES: list[str] = [
    col for col in SCHEMA
    if col not in (PERIOD_COL, TARGET)
]

# mths_since columns: fill null with sentinel (null = event never occurred)
ENG_FILL_MTHS: list[str] = [
    "mths_since_recent_bc",
    "mths_since_recent_inq",
    "mths_since_rcnt_il",
    "emp_length",
]

# columns to add binary null indicator before any imputation
ENG_NULL_MTHS: list[str] = [
    "mths_since_recent_inq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "emp_length",
]

# columns to drop after feature engineering
ENG_DROP: list[str] = [
    "mths_since_recent_inq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "earliest_cr_line",
]

# derived column from earliest_cr_line
ENG_CR_LINE = "cr_age_mths"

# columns the imputer should skip entirely
IMPUTER_SKIP: set[str] = set(ENG_FILL_MTHS + ENG_DROP + [
    "earliest_cr_line",
    TARGET,
    TARGET_ENG,
    PERIOD_COL,
])

# final feature set after all engineering
FINAL_FEATURES: list[str] = (
    [col for col in KEEP_FEATURES if col not in ENG_DROP]
    + [f"{col}_null" for col in ENG_NULL_MTHS]
    + [ENG_CR_LINE]
)