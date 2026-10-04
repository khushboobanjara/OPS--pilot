"""Application service: runs the pipeline AND persists the outcome, plus the human-review
workflow, audit trail and metrics. The API layer stays a thin wrapper around this."""
from collections import defaultdict

from sqlalchemy import func

from app.db import AuditLog, Email, Invoice, SessionLocal, utcnow
from app.pipeline.base import PipelineContext
from app.review.rules import EDITABLE_FIELDS, ReviewError, plan_review, status_for_decision

# Cost-model assumptions for /metrics (estimates, not measurements)
MANUAL_MINUTES_PER_INVOICE = 6.0
HOURLY_COST = 30.0


class NotFound(Exception):
    pass


def _iso(dt):
    return dt.isoformat() if dt else None


def invoice_to_dict(inv: Invoice, email: Email | None = None) -> dict:
    d = {
        "id": inv.id, "email_id": inv.email_id,
        "vendor_name": inv.vendor_name, "invoice_number": inv.invoice_number,
        "invoice_date": inv.invoice_date, "due_date": inv.due_date, "currency": inv.currency,
        "total_amount": inv.total_amount, "po_number": inv.po_number,
        "extraction_confidence": inv.extraction_confidence,
        "risk_score": inv.risk_score, "risk_breakdown": inv.risk_breakdown,
        "decision": inv.decision, "decision_rule": inv.decision_rule, "decision_reasons": inv.decision_reasons,
        "status": inv.status, "duplicate_of": inv.duplicate_of, "flags": inv.flags,
        "reviewed_by": inv.reviewed_by, "review_comment": inv.review_comment, "reviewed_at": _iso(inv.reviewed_at),
        "created_at": _iso(inv.created_at),
    }
    if email is not None:
        d["email"] = {"subject": email.subject, "sender": email.sender, "body": email.body}
    return d


# ---------------------------------------------------------------- ingestion
def process_email(pipeline, raw: dict) -> dict:
    """Idempotent on message_id: re-sending the same email returns the stored result."""
    with SessionLocal() as s:
        email = s.query(Email).filter_by(message_id=raw["message_id"]).one_or_none()
        if email is not None:
            inv = s.query(Invoice).filter_by(email_id=email.id).one_or_none()
            if inv is not None or email.category is not None:
                return {"email_id": email.id, "message_id": email.message_id, "category": email.category,
                        "category_confidence": email.category_confidence,
                        "decision": inv.decision if inv else "not_invoice",
                        "invoice": invoice_to_dict(inv) if inv else None,
                        "flags": inv.flags if inv else [], "already_processed": True}
            email_id = email.id            # stored earlier but never finished: process it now
            received_at = email.received_at
        else:
            email = Email(message_id=raw["message_id"], sender=raw.get("sender", ""),
                          subject=raw.get("subject", ""), body=raw.get("body", ""))
            s.add(email)
            s.commit()
            email_id = email.id
            received_at = email.received_at

    # Pipeline runs outside the session; it opens its own short sessions for audit rows.
    ctx = pipeline.run(PipelineContext(
        email={"subject": raw.get("subject", ""), "body": raw.get("body", ""), "sender": raw.get("sender", ""),
               "received_at": received_at},
        email_id=email_id))

    with SessionLocal() as s:
        email = s.get(Email, email_id)
        email.category = ctx.email.get("category")
        email.category_confidence = ctx.email.get("category_confidence")

        invoice = None
        if ctx.decision != "not_invoice":
            invoice = _store_invoice(s, email_id, ctx)
        s.commit()
        return {"email_id": email_id, "message_id": email.message_id, "category": email.category,
                "category_confidence": email.category_confidence,
                "decision": invoice.decision if invoice else ctx.decision,
                "invoice": invoice_to_dict(invoice) if invoice else None,
                "flags": list(ctx.flags), "already_processed": False}


def _store_invoice(s, email_id: int, ctx: PipelineContext) -> Invoice:
    inv = ctx.invoice
    # Empty invoice dict = classifier was unsure, so nothing was extracted. We still create a row
    # so the email lands in the review queue instead of silently disappearing.
    outcome = ctx.decision or "human_review"                      # no decision made -> fail safe
    rule = inv.get("decision_rule") or ("low_classification_confidence" if not inv else "no_decision_made")
    reasons = inv.get("decision_reasons") or list(ctx.flags)
    dup = inv.get("duplicate_of")

    row = Invoice(
        email_id=email_id,
        vendor_name=inv.get("vendor_name"), invoice_number=inv.get("invoice_number"),
        invoice_date=inv.get("invoice_date"), due_date=inv.get("due_date"), currency=inv.get("currency"),
        total_amount=inv.get("total_amount"), po_number=inv.get("po_number"),
        extraction_confidence=inv.get("extraction_confidence"),
        risk_score=inv.get("risk_score"), risk_breakdown=inv.get("risk_breakdown"),
        decision=outcome, status=status_for_decision(outcome), decision_rule=rule,
        decision_reasons=reasons, flags=list(ctx.flags),
        duplicate_of=dup if isinstance(dup, int) else None,
    )
    s.add(row)
    s.flush()                                                     # assigns row.id
    s.add(AuditLog(entity_type="invoice", entity_id=row.id, stage="decision", action=outcome, actor="system",
                   details={"rule": rule, "risk_score": row.risk_score, "reasons": reasons,
                            "risk_breakdown": row.risk_breakdown, "status": row.status}))
    return row


# ---------------------------------------------------------------- queries
def review_queue(limit: int = 50) -> list[dict]:
    with SessionLocal() as s:
        # Unknown risk (NULL) sorts as highest priority.
        rows = (s.query(Invoice).filter(Invoice.status == "pending_review")
                .order_by(func.coalesce(Invoice.risk_score, 1.0).desc(), Invoice.created_at).limit(limit).all())
        return [invoice_to_dict(r, s.get(Email, r.email_id)) for r in rows]


def list_invoices(status: str | None = None, decision: str | None = None, limit: int = 100) -> list[dict]:
    with SessionLocal() as s:
        q = s.query(Invoice)
        if status:
            q = q.filter(Invoice.status == status)
        if decision:
            q = q.filter(Invoice.decision == decision)
        return [invoice_to_dict(r) for r in q.order_by(Invoice.id.desc()).limit(limit).all()]


def get_invoice(invoice_id: int) -> dict:
    with SessionLocal() as s:
        inv = s.get(Invoice, invoice_id)
        if inv is None:
            raise NotFound(f"Invoice {invoice_id} not found")
        return invoice_to_dict(inv, s.get(Email, inv.email_id))


def get_audit(invoice_id: int) -> list[dict]:
    """Full story of one invoice: every pipeline stage, the system decision, and human actions."""
    with SessionLocal() as s:
        inv = s.get(Invoice, invoice_id)
        if inv is None:
            raise NotFound(f"Invoice {invoice_id} not found")
        rows = (s.query(AuditLog)
                .filter(((AuditLog.entity_type == "invoice") & (AuditLog.entity_id == invoice_id)) |
                        ((AuditLog.entity_type == "email") & (AuditLog.entity_id == inv.email_id)))
                .order_by(AuditLog.timestamp, AuditLog.id).all())
        return [{"id": r.id, "timestamp": _iso(r.timestamp), "entity": f"{r.entity_type}:{r.entity_id}",
                 "stage": r.stage, "action": r.action, "actor": r.actor,
                 "duration_ms": r.duration_ms, "details": r.details} for r in rows]


# ---------------------------------------------------------------- human review
def review_invoice(invoice_id: int, action: str, reviewer: str, comment: str = "",
                   corrections: dict | None = None) -> dict:
    with SessionLocal() as s:
        inv = s.get(Invoice, invoice_id)
        if inv is None:
            raise NotFound(f"Invoice {invoice_id} not found")

        candidate = {f: getattr(inv, f) for f in EDITABLE_FIELDS}
        changes = {}
        for k, v in (corrections or {}).items():
            if k not in EDITABLE_FIELDS:
                raise ReviewError(f"Field {k!r} cannot be corrected", "invalid")
            if isinstance(v, str):
                v = v.strip().upper() if k == "currency" else v.strip()
            if v != candidate[k]:
                changes[k] = {"from": candidate[k], "to": v}
                candidate[k] = v

        new_status = plan_review(inv.status, action, candidate)   # may raise; nothing is modified yet

        previous_status = inv.status
        for k, ch in changes.items():
            setattr(inv, k, ch["to"])
        inv.status, inv.reviewed_by, inv.review_comment, inv.reviewed_at = new_status, reviewer, comment, utcnow()
        s.add(AuditLog(entity_type="invoice", entity_id=inv.id, stage="human_review", action=new_status,
                       actor=reviewer,
                       details={"comment": comment, "corrections": changes, "previous_status": previous_status,
                                "system_decision": inv.decision, "risk_score": inv.risk_score}))
        s.commit()
        return invoice_to_dict(inv, s.get(Email, inv.email_id))


# ---------------------------------------------------------------- metrics
def metrics() -> dict:
    with SessionLocal() as s:
        total = s.query(func.count(Invoice.id)).scalar() or 0
        by_decision = dict(s.query(Invoice.decision, func.count(Invoice.id)).group_by(Invoice.decision).all())
        by_status = dict(s.query(Invoice.status, func.count(Invoice.id)).group_by(Invoice.status).all())
        emails = s.query(func.count(Email.id)).scalar() or 0

        per_email = defaultdict(float)
        for entity_id, ms in s.query(AuditLog.entity_id, AuditLog.duration_ms).filter(
                AuditLog.entity_type == "email", AuditLog.duration_ms.isnot(None)).all():
            per_email[entity_id] += ms

    auto = by_decision.get("auto_approved", 0)
    hours = auto * MANUAL_MINUTES_PER_INVOICE / 60
    return {
        "emails_received": emails, "invoices": total,
        "by_decision": by_decision, "by_status": by_status,
        "automation_rate": round(auto / total, 4) if total else None,
        "pending_review": by_status.get("pending_review", 0),
        "avg_pipeline_ms_per_email": round(sum(per_email.values()) / len(per_email), 2) if per_email else None,
        "estimated_savings": {
            "hours": round(hours, 2), "cost": round(hours * HOURLY_COST, 2),
            "assumptions": f"{MANUAL_MINUTES_PER_INVOICE:g} min manual effort per invoice at {HOURLY_COST:g}/hour",
            "note": "estimate, not a measurement"},
    }
