"""Replay the synthetic emails in order and measure duplicate detection against ground truth."""
import json
from pathlib import Path

from app.duplicates.matcher import find_duplicate
from app.extraction.regex_extractor import extract_fields

rows = [json.loads(l) for l in (Path(__file__).resolve().parent.parent / "data" / "synthetic_emails.jsonl").read_text().splitlines()]
history, tp = [], 0
fp = fn = tn = 0
for r in rows:
    if r["category"] != "invoice":
        continue
    f = extract_fields(r["subject"], r["body"])
    inv = {k: (v.value if v else None) for k, v in f.items()}
    m = find_duplicate(inv, history)
    flagged = m is not None
    if flagged and r["is_duplicate"]: tp += 1
    elif flagged: fp += 1; print("FALSE POSITIVE", r["message_id"], m)
    elif r["is_duplicate"]: fn += 1; print("MISSED", r["message_id"])
    else: tn += 1
    history.append({"id": r["message_id"], **inv})

prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}  precision={prec:.1%} recall={rec:.1%}")
