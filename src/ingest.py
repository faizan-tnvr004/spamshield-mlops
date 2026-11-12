"""Download the UCI SMS Spam Collection and write data/raw/sms.csv."""
import io, zipfile
from pathlib import Path
import pandas as pd, requests, yaml

def main():
    p = yaml.safe_load(open("params.yaml"))["ingest"]
    r = requests.get(p["url"], timeout=60); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open("SMSSpamCollection"), sep="\t", header=None, names=["label", "text"])
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/sms.csv", index=False)
    print(f"ingested {len(df)} rows")

if __name__ == "__main__":
    main()
