from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

ARTIFACT_DIR = PROJECT_ROOT / "artifact"
ARTIFACT_PREPROCESS_DIR = ARTIFACT_DIR / "preprocessing"

RAW_DATA_FILE = RAW_DIR / "lending_club" / "Loan_status_2007-2020Q3.gzip"

def validate_paths() -> None:
    paths = {
        "PROJECT_ROOT":                 PROJECT_ROOT,
        "DATA_DIR":                     DATA_DIR,
        "RAW_DIR":                      RAW_DIR,
        "INTERIM_DIR":                  INTERIM_DIR,
        "PROCESSED_DIR":                PROCESSED_DIR,
        
        "ARTIFACT_DIR" :                ARTIFACT_DIR,
        "ARTIFACT_PREPROCESS_DIR" :     ARTIFACT_PREPROCESS_DIR,

        "RAW_DATA_FILE":                RAW_DATA_FILE,
    }
    for name, path in paths.items():
        status = "✓" if path.exists() else "✗ NOT FOUND"
        print(f"{status}  {name}: {path}")