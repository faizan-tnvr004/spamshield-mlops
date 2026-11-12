"""Schema / null / range checks. Exits non-zero on failure; writes reports/validation.json."""
import json, sys
from pathlib import Path
import pandas as pd

def main():
    df = pd.read_csv("data/raw/sms.csv")
    checks = {
        "schema": bool(list(df.columns) == ["label", "text"]),
        "no_nulls": bool(int(df.isna().sum().sum()) == 0),
        "labels_valid": bool(set(df["label"].unique()) <= {"ham", "spam"}),
        "text_length_range": bool(df["text"].str.len().between(1, 1000).all()),
        "min_rows": bool(len(df) >= 5000),
        "spam_ratio_range": bool(0.05 <= (df["label"] == "spam").mean() <= 0.30),
    }
    Path("reports").mkdir(exist_ok=True)
    json.dump(checks, open("reports/validation.json", "w"), indent=2)
    bad = [k for k, v in checks.items() if not v]
    if bad:
        sys.exit(f"validation failed: {bad}")
    print("validation passed", checks)

if __name__ == "__main__":
    main()
