from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Email(Base):
    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[str] = mapped_column(String(200), unique=True)
    sender: Mapped[str] = mapped_column(String(200))
    subject: Mapped[str] = mapped_column(String(500))
    body: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    category: Mapped[str | None] = mapped_column(String(50))
    category_confidence: Mapped[float | None] = mapped_column(Float)


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"))
    vendor_name: Mapped[str | None] = mapped_column(String(200))
    invoice_number: Mapped[str | None] = mapped_column(String(100))
    invoice_date: Mapped[str | None] = mapped_column(String(20))
    due_date: Mapped[str | None] = mapped_column(String(20))
    currency: Mapped[str | None] = mapped_column(String(3))
    total_amount: Mapped[float | None] = mapped_column(Float)
    po_number: Mapped[str | None] = mapped_column(String(100))
    extraction_confidence: Mapped[float | None] = mapped_column(Float)
    risk_score: Mapped[float | None] = mapped_column(Float)
    decision: Mapped[str | None] = mapped_column(String(30))  # auto_approved | human_review | rejected
    # `decision` is the SYSTEM's outcome and never changes.
    # `status` is the lifecycle: approved | pending_review | rejected (humans move pending_review -> approved/rejected)
    status: Mapped[str] = mapped_column(String(30), default="new")
    duplicate_of: Mapped[int | None] = mapped_column(Integer)
    decision_rule: Mapped[str | None] = mapped_column(String(60))
    decision_reasons: Mapped[list | None] = mapped_column(JSON)
    flags: Mapped[list | None] = mapped_column(JSON)
    risk_breakdown: Mapped[dict | None] = mapped_column(JSON)
    reviewed_by: Mapped[str | None] = mapped_column(String(100))
    review_comment: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(30))
    entity_id: Mapped[int | None]
    stage: Mapped[str] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(100))
    actor: Mapped[str] = mapped_column(String(50), default="system")
    details: Mapped[dict | None] = mapped_column(JSON)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)
