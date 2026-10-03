"""Train the email classifier.   Run:  python -m app.ml.train_classifier"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import settings

DATA = Path("data/synthetic_emails.jsonl")


def email_text(e: dict) -> str:
    # Subject is repeated so it carries more weight than the body.
    return f"{e['subject']} {e['subject']} {e['body']}"


def main() -> None:
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    X = [email_text(r) for r in rows]
    y = [r["category"] for r in rows]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_tr, y_tr)

    pred = model.predict(X_te)
    print(classification_report(y_te, pred, digits=3))
    labels = list(model.classes_)
    print("Labels:", labels)
    print(confusion_matrix(y_te, pred, labels=labels))

    # How does confidence relate to correctness? This is what justifies the threshold.
    proba = model.predict_proba(X_te)
    conf, correct = proba.max(axis=1), pred == np.array(y_te)
    t = settings.classifier_min_confidence
    kept = conf >= t
    print(f"\nThreshold {t}: auto-handles {kept.mean():.1%} of emails, "
          f"accuracy on those = {correct[kept].mean():.1%}, "
          f"sent to review = {(~kept).mean():.1%}")

    Path(settings.classifier_model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.classifier_model_path)
    print(f"Saved model to {settings.classifier_model_path}")


if __name__ == "__main__":
    main()
