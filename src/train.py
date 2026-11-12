"""Train baseline (logreg) + other models, log all runs to MLflow, register best/main model."""
import json, os
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, pandas as pd, yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def build(name, p, seed):
    tf = TfidfVectorizer(max_features=p["tfidf"]["max_features"], ngram_range=tuple(p["tfidf"]["ngram_range"]))
    m = p["models"][name]
    clf = {"logreg": lambda: LogisticRegression(random_state=seed, **m),
           "naive_bayes": lambda: MultinomialNB(**m),
           "random_forest": lambda: RandomForestClassifier(random_state=seed, n_jobs=-1, **m)}[name]()
    return Pipeline([("tfidf", tf), ("clf", clf)]), m

def main():
    p = yaml.safe_load(open("params.yaml"))
    mp = p["mlflow"]
    # Remote tracking: set MLFLOW_TRACKING_URI (+ MLFLOW_TRACKING_USERNAME/PASSWORD) for DagsHub.
    # Falls back to local ./mlruns only if unset (for offline dev).
    mlflow.set_experiment(mp["experiment"])
    tr, te = pd.read_csv("data/processed/train.csv"), pd.read_csv("data/processed/test.csv")
    results = {}
    for name in ["logreg", "naive_bayes", "random_forest"]:
        pipe, params = build(name, p, p["seed"])
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("role", "baseline" if name == "logreg" else "main-candidate")
            mlflow.log_params({"model": name, **params, **{f"tfidf_{k}": str(v) for k, v in p["tfidf"].items()}})
            pipe.fit(tr["clean"], tr["y"])
            pred, proba = pipe.predict(te["clean"]), pipe.predict_proba(te["clean"])[:, 1]
            met = {"accuracy": accuracy_score(te["y"], pred), "precision": precision_score(te["y"], pred),
                   "recall": recall_score(te["y"], pred), "f1": f1_score(te["y"], pred),
                   "roc_auc": roc_auc_score(te["y"], proba)}
            mlflow.log_metrics(met)
            mlflow.log_artifact("params.yaml")
            mlflow.sklearn.log_model(pipe, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])  # our own model
            results[name] = {**met, "run_id": run.info.run_id, "pipe": pipe}
            print(name, {k: round(v, 4) for k, v in met.items()})
    best = max(results, key=lambda k: results[k]["f1"])
    main_name = mp["main_model"] if mp["main_model"] in results else best
    Path("models").mkdir(exist_ok=True)
    joblib.dump(results[main_name]["pipe"], "models/model.joblib")
    Path("reports").mkdir(exist_ok=True)
    json.dump({k: {m: v for m, v in r.items() if m not in ("pipe",)} for k, r in results.items()} | {"selected": best},
              open("reports/metrics.json", "w"), indent=2)
    # Register best-F1 model and move to Staging
    client = MlflowClient()
    name = mp["registered_model"]
    try:
        client.create_registered_model(name)
    except Exception:
        pass  # already exists
    # DagsHub lacks MLflow-3 logged-model lookup, so register from the run's artifact URI directly.
    run_id = results[best]["run_id"]
    src = client.get_run(run_id).info.artifact_uri + "/model"
    mv = client.create_model_version(name, src, run_id)
    try:
        client.transition_model_version_stage(mp["registered_model"], mv.version, "Staging")
    except Exception:  # stages deprecated in newer MLflow; alias is set below
        pass
    client.set_registered_model_alias(name, "staging", mv.version)  # DagsHub UI shows aliases, not stages
    client.set_model_version_tag(name, mv.version, "stage", "Staging")
    print(f"registered {best} as {mp['registered_model']} v{mv.version} -> Staging")

if __name__ == "__main__":
    main()
