"""End-to-end replay of the synthetic invoices through every stage up to the decision,
scored against ground truth. Also sweeps the auto-approve threshold so you can tune it.
Run:  python -m scripts.evaluate_decisions

Ground truth: an invoice is 'bad' if it is a duplicate or an anomaly. The key metrics:
  automation rate  - share of invoices approved with no human touch (want HIGH)
  leakage          - bad invoices that got auto-approved (want ZERO)
  false alarms     - good invoices that still went to a human (want LOW)"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.pipeline.anomaly import AnomalyStage
from app.pipeline.base import PipelineContext
from app.pipeline.duplicates import DuplicateStage
from app.pipeline.extract import ExtractStage
from app.pipeline.risk import RiskStage
from app.pipeline.validate import ValidateStage
from app.risk.decision import Thresholds, decide

DATA = Path("data/synthetic_emails.jsonl")
# Cost model assumptions (edit to match your story). Printed with the results.
MINUTES_PER_MANUAL_INVOICE = 6
HOURLY_COST = 30.0


def replay():
    rows = [json.loads(l) for l in DATA.read_text().splitlines() if l.strip()]
    inv_history: list[dict] = []
    amounts: dict[str, list[float]] = defaultdict(list)
    stages = [ExtractStage(), ValidateStage(),
              DuplicateStage(history_provider=lambda inv: inv_history),
              AnomalyStage(history_provider=lambda v: amounts[v]),
              RiskStage()]
    out = []
    for r in rows:
        if r["category"] != "invoice":
            continue
        ctx = PipelineContext(email={"subject": r["subject"], "body": r["body"]})
        for s in stages:
            ctx = s.run(ctx)
        inv = ctx.invoice
        inv_history.append({"id": r["message_id"], **{k: inv.get(k) for k in
                            ("vendor_name", "invoice_number", "invoice_date", "total_amount")}})
        if inv.get("total_amount") is not None and "duplicate_of" not in inv:
            amounts[inv.get("vendor_name")].append(inv["total_amount"])
        out.append(dict(bad=r["is_duplicate"] or r["is_anomaly"], dup=r["is_duplicate"], anom=r["is_anomaly"],
                        invoice=inv, signals=dict(ctx.risk_signals),
                        score=inv["risk_score"], contrib=inv["risk_breakdown"]))
    return out


def evaluate(results, t: Thresholds):
    decisions = [(x, decide(x["invoice"], x["signals"], x["score"], x["contrib"], t)) for x in results]
    n = len(decisions)
    auto = [x for x, d in decisions if d.outcome == "auto_approved"]
    leak = [x for x in auto if x["bad"]]
    good = [(x, d) for x, d in decisions if not x["bad"]]
    false_alarm = [(x, d) for x, d in good if d.outcome != "auto_approved"]
    return decisions, auto, leak, good, false_alarm, n


def main() -> None:
    results = replay()
    t = Thresholds()
    decisions, auto, leak, good, false_alarm, n = evaluate(results, t)

    print(f"{n} invoices replayed | thresholds: approve<={t.auto_approve_max_risk} reject>={t.auto_reject_min_risk} "
          f"amount cap {t.auto_approve_max_amount:,.0f}\n")
    print("Decisions:", dict(Counter(d.outcome for _, d in decisions)))
    print(f"Automation rate:      {len(auto)/n:.1%}  ({len(auto)} of {n} approved with no human touch)")
    bad_total = sum(x['bad'] for x in results)
    print(f"Leakage:              {len(leak)} of {bad_total} bad invoices were auto-approved")
    for x in leak[:5]:
        print("   LEAKED:", x["invoice"].get("invoice_number"), x["invoice"].get("total_amount"), "dup" if x["dup"] else "anomaly")
    print(f"False alarms:         {len(false_alarm)} of {len(good)} good invoices went to a human or were rejected "
          f"({len(false_alarm)/len(good):.1%})")
    print("Rules behind false alarms:", dict(Counter(d.rule for _, d in false_alarm)))
    dup_rej = sum(1 for x, d in decisions if x["dup"] and d.outcome == "rejected")
    dup_rev = sum(1 for x, d in decisions if x["dup"] and d.outcome == "human_review")
    print(f"Duplicates:           {dup_rej} rejected, {dup_rev} sent to review, of {sum(x['dup'] for x in results)}")
    print(f"Anomalies:            {sum(1 for x, d in decisions if x['anom'] and d.outcome == 'human_review')} sent to review, "
          f"of {sum(x['anom'] for x in results)}")

    hours = len(auto) * MINUTES_PER_MANUAL_INVOICE / 60
    print(f"\nEstimated saving: {len(auto)} auto-approved x {MINUTES_PER_MANUAL_INVOICE} min = {hours:.1f} h "
          f"= ${hours*HOURLY_COST:,.0f} at ${HOURLY_COST:.0f}/h  (assumptions, not measured)")

    print("\nThreshold sweep (auto_approve_max_risk):")
    print("  threshold  automation  leakage  false_alarm_rate")
    for thr in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
        _, a, l, g, fa, nn = evaluate(results, Thresholds(auto_approve_max_risk=thr))
        print(f"  {thr:<9}  {len(a)/nn:>9.1%}  {len(l):>7}  {len(fa)/len(g):>15.1%}")


if __name__ == "__main__":
    main()
