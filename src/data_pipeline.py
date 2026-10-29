"""Load raw German Credit records into SQLite and build a clean, reusable modelling table with SQL."""
import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "german_credit_data.csv"
DB_PATH = ROOT / "data" / "credit.db"

CLEAN_SQL = """
DROP TABLE IF EXISTS loans_clean;
CREATE TABLE loans_clean AS
SELECT
    loan_id,
    age,
    sex,
    job,
    housing,
    COALESCE(NULLIF(saving_accounts, 'NA'), 'unknown')   AS saving_accounts,
    COALESCE(NULLIF(checking_account, 'NA'), 'unknown')  AS checking_account,
    credit_amount,
    duration,
    purpose,
    CAST(credit_amount AS REAL) / duration               AS monthly_burden,
    CASE WHEN age < 25 THEN '<25'
         WHEN age < 35 THEN '25-34'
         WHEN age < 50 THEN '35-49'
         ELSE '50+' END                                  AS age_band,
    CASE WHEN risk = 'bad' THEN 1 ELSE 0 END             AS is_default
FROM loans_raw
WHERE credit_amount > 0 AND duration > 0;
"""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset() -> pd.DataFrame:
    raw = pd.read_csv(RAW_CSV, index_col=0, keep_default_na=False)
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    raw.index.name = "loan_id"

    with sqlite3.connect(DB_PATH) as conn:
        raw.to_sql("loans_raw", conn, if_exists="replace")
        conn.executescript(CLEAN_SQL)
        df = pd.read_sql("SELECT * FROM loans_clean", conn)
    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"{len(df)} rows, default rate {df.is_default.mean():.1%}, source sha256 {file_sha256(RAW_CSV)[:12]}")
