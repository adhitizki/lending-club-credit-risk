# src/utils/logger.py
import time
import logging
import traceback
from contextlib import contextmanager
from datetime import datetime
import polars as pl


# ── logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname).1s %(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("pipeline")


# ── helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def info(msg: str) -> None:
    print(f"[I {_now()}] {msg}")


def success(msg: str) -> None:
    print(f"[S {_now()}] {msg}")


def warn(msg: str) -> None:
    print(f"[W {_now()}] {msg}")


def error(msg: str) -> None:
    print(f"[E {_now()}] {msg}")


def shape(df: pl.LazyFrame) -> tuple[int, int]:
    return df.select(pl.len()).collect().item(), len(df.collect_schema())


def log_shape(label: str, df: pl.LazyFrame) -> None:
    rows, cols = shape(df)
    info(f"  {label:<22} rows: {rows:>10,}   cols: {cols:>4}")


def fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m {s:.2f}s"


def header(title: str, width: int = 60) -> None:
    print(f"\n{'#'*width}")
    info(title)
    print(f"{'#'*width}")


def divider(title: str, width: int = 60) -> None:
    print(f"{'='*width}")
    info(title)
    print(f"{'='*width}")


@contextmanager
def step(step_num: int, title: str, width: int = 60):
    print(f"\n{'='*width}")
    info(f"STEP {step_num}: {title} — started")
    print(f"{'='*width}")
    start = time.time()
    try:
        yield
        elapsed = fmt_time(time.time() - start)
        success(f"STEP {step_num}: {title} — completed in {elapsed}")
    except Exception as e:
        elapsed = fmt_time(time.time() - start)
        error(f"STEP {step_num}: {title} — failed after {elapsed}")
        error(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        raise