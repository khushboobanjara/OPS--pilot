from pathlib import Path
from typing import Callable

from app.anomaly.detector import AmountAnomalyDetector
from app.anomaly.scoring import score_amount
from app.config import settings
from app.pipeline.base import PipelineContext, Stage

AmountHistoryProvider = Callable[[str | None], list[float]]
COLD_START_SIGNAL = 0.15   # "unknown" is mildly risky, not safe


def db_amount_history(vendor: str | None) -> list[float]:
    from app.db import Invoice, SessionLocal
    with SessionLocal() as s:
        # Only approved, non-duplicate invoices define what is "normal" for a vendor:
        # pending or rejected anomalies must not teach the model that anomalies are normal.
        rows = (s.query(Invoice)
                .filter(Invoice.vendor_name == vendor, Invoice.status == "approved", Invoice.duplicate_of.is_(None))
                .all())
        return [r.total_amount for r in rows if r.total_amount is not None]


class AnomalyStage(Stage):
    name = "anomaly"

    def __init__(self, history_provider: AmountHistoryProvider = db_amount_history, model_path: str | None = None):
        self.history_provider = history_provider
        path = model_path or settings.anomaly_model_path
        self.detector = AmountAnomalyDetector.load(path) if Path(path).exists() else None

    def run(self, ctx: PipelineContext) -> PipelineContext:
        amount, vendor = ctx.invoice.get("total_amount"), ctx.invoice.get("vendor_name")
        if amount is None:
            ctx.risk_signals["anomaly"] = 0.0      # missing amount is already flagged upstream
            return ctx
        signal, reasons = score_amount(amount, self.history_provider(vendor), self.detector)
        if signal is None:
            signal = COLD_START_SIGNAL
        ctx.risk_signals["anomaly"] = signal
        ctx.flags.extend(f"Anomaly: {r}" for r in reasons)
        return ctx
