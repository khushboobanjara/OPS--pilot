"""Train + evaluate the anomaly detector.   Run:  python -m app.ml.train_anomaly

Replays invoices in arrival order. Each invoice is featurized against its vendor's
PRIOR invoices only, then we split chronologically (train on the first 70%, test on
the last 30%), which mimics deploying the model on future data."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.anomaly.detector import AmountAnomalyDetector
from app.anomaly.features import MIN_HISTORY, featurize
from app.config import settings
from app.extraction.regex_extractor import extract_fields

DATA = Path("data/synthetic_emails.jsonl")
Z_THRESHOLD = 4.0   # baseline rule: |robust z| above this = anomaly


def build_dataset():
    rows = [json.loads(l) for l in DATA.read_text().splitlines()]
    history = defaultdict(list)
    X, y, cold = [], [], 0
    for r in rows:
        if r["category"] != "invoice" or r["is_duplicate"]:
            continue                                 # duplicates are Phase 4's job
        f = extract_fields(r["subject"], r["body"])
        vendor, amount = f["vendor_name"].value, f["total_amount"].value
        feats = featurize(amount, history[vendor])
        history[vendor].append(amount)
        if feats is None:
            cold += 1
            continue
        X.append(feats); y.append(int(r["is_anomaly"]))
    return np.array(X), np.array(y), cold


def prf(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, p, r


def main() -> None:
    X, y, cold = build_dataset()
    cut = int(len(X) * 0.7)
    X_tr, X_te, y_tr, y_te = X[:cut], X[cut:], y[:cut], y[cut:]
    print(f"{len(X)} scored invoices ({cold} skipped: vendor cold start < {MIN_HISTORY} prior invoices)")
    print(f"train={len(X_tr)} test={len(X_te)} | anomalies in test: {int(y_te.sum())}\n")

    # Baseline: plain statistical rule on the robust z-score
    tp, fp, fn, p, r = prf(y_te, (np.abs(X_te[:, 1]) > Z_THRESHOLD).astype(int))
    print(f"Baseline |robust z| > {Z_THRESHOLD}:  TP={tp} FP={fp} FN={fn}  precision={p:.1%} recall={r:.1%}")

    det = AmountAnomalyDetector().fit(X_tr)
    sig = np.array([det.signal(x) for x in X_te])
    for thr in (0.90, 0.95, 0.97):
        tp, fp, fn, p, r = prf(y_te, (sig >= thr).astype(int))
        print(f"IsolationForest signal >= {thr}:  TP={tp} FP={fp} FN={fn}  precision={p:.1%} recall={r:.1%}")

    # Final model uses all data
    Path(settings.anomaly_model_path).parent.mkdir(parents=True, exist_ok=True)
    AmountAnomalyDetector().fit(X).save(settings.anomaly_model_path)
    print(f"\nSaved model -> {settings.anomaly_model_path}")


if __name__ == "__main__":
    main()
