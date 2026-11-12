# SpamShield — Reproducible ML Pipeline (Milestone 1)

SMS spam detection pipeline: DVC-versioned data, validated, trained with a baseline (Logistic Regression) and
two further models (Naive Bayes, Random Forest), tracked in MLflow and registered in the Model Registry (Staging).
No Docker / cloud / API / monitoring yet (later milestones).

Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,574 messages).

## Setup
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

### Remote configuration (DagsHub) — credentials stay out of git
```bash
# DVC remote
dvc remote add -d origin https://dagshub.com/<user>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <dagshub-user>
dvc remote modify origin --local password <dagshub-token>
# MLflow remote
export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<dagshub-user>
export MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```
If `MLFLOW_TRACKING_URI` is unset, MLflow logs to local `./mlruns` (dev only).

## Run
```bash
dvc repro        # ingest -> validate -> preprocess -> train (all stages)
dvc push         # upload data/models to the DVC remote
dvc dag          # pipeline diagram
```
Everything is parameterised in `params.yaml`; changing a parameter re-runs only affected stages.

## Pipeline stages (`dvc.yaml`)
| Stage | Script | Output |
|---|---|---|
| ingest | `src/ingest.py` | `data/raw/sms.csv` (downloaded from UCI) |
| validate | `src/validate.py` | `reports/validation.json` (schema, nulls, label set, text length range, row count, spam ratio); fails the pipeline on error |
| preprocess | `src/preprocess.py` | `data/processed/{train,test}.csv` (clean, dedupe, stratified split) |
| train | `src/train.py` | `models/model.joblib`, `reports/metrics.json`; logs 3 MLflow runs, registers best-F1 model → Staging |

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.9845 | 0.9915 | 0.8855 | 0.9355 |
| Naive Bayes (main) | 0.9903 | 0.9840 | 0.9389 | 0.9609 |
| Random Forest | 0.9884 | 1.0000 | 0.9084 | 0.9520 |

## Structure
```
dvc.yaml / dvc.lock   pipeline definition and locked hashes
params.yaml           all hyperparameters and config
src/                  ingest, validate, preprocess, train
data/                 raw + processed data (DVC-tracked, not in git)
models/               model weights (DVC-tracked, not in git)
reports/              metrics and validation output
```
