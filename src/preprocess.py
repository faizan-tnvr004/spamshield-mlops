"""Clean text, deduplicate and split into train/test."""
import re
from pathlib import Path
import pandas as pd, yaml
from sklearn.model_selection import train_test_split

def clean(t):
    t = re.sub(r"http\S+", " url ", t.lower())
    t = re.sub(r"\d+", " num ", t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t)).strip()

def main():
    p = yaml.safe_load(open("params.yaml"))
    df = pd.read_csv("data/raw/sms.csv").drop_duplicates()
    df["clean"] = df["text"].map(clean)
    df = df[df["clean"] != ""].copy()
    df["y"] = (df["label"] == "spam").astype(int)
    tr, te = train_test_split(df, test_size=p["split"]["test_size"], stratify=df["y"], random_state=p["seed"])
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    tr.to_csv("data/processed/train.csv", index=False)
    te.to_csv("data/processed/test.csv", index=False)
    print(len(tr), len(te))

if __name__ == "__main__":
    main()
