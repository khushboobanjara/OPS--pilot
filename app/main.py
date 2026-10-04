"""OpsPilot AI HTTP API.   Run:  uvicorn app.main:app --reload
Dashboard: http://127.0.0.1:8000/     API docs: http://127.0.0.1:8000/docs"""
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.db import init_db
from app.pipeline.factory import build_pipeline
from app.review.rules import ReviewError
from app.services import processing


class EmailIn(BaseModel):
    message_id: str = Field(min_length=1)
    sender: str = ""
    subject: str = ""
    body: str = ""
    received_at: datetime | None = None    # when the mail server received it; defaults to "now"


class Corrections(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    total_amount: float | None = None
    po_number: str | None = None


class ReviewIn(BaseModel):
    action: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=100)
    comment: str = ""
    corrections: Corrections | None = None


def create_app(pipeline_factory: Callable = build_pipeline) -> FastAPI:
    """Factory so tests can inject a lighter pipeline (e.g. without the ML classifier)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db()
        app.state.pipeline = pipeline_factory()
        yield

    app = FastAPI(title="OpsPilot AI", version="0.7.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(Path(__file__).parent / "static" / "dashboard.html")

    @app.post("/emails")
    def post_email(payload: EmailIn, request: Request):
        """Run an incoming email through the full pipeline and store the outcome."""
        return processing.process_email(request.app.state.pipeline, payload.model_dump())

    @app.get("/review-queue")
    def get_review_queue(limit: int = Query(50, ge=1, le=500)):
        """Invoices waiting for a human, highest risk first."""
        return processing.review_queue(limit)

    @app.get("/invoices")
    def get_invoices(status: str | None = None, decision: str | None = None, limit: int = Query(100, ge=1, le=1000)):
        return processing.list_invoices(status, decision, limit)

    @app.get("/invoices/{invoice_id}")
    def get_invoice(invoice_id: int):
        try:
            return processing.get_invoice(invoice_id)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))

    @app.get("/invoices/{invoice_id}/audit")
    def get_invoice_audit(invoice_id: int):
        try:
            return processing.get_audit(invoice_id)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))

    @app.post("/invoices/{invoice_id}/review")
    def review_invoice(invoice_id: int, payload: ReviewIn):
        try:
            corrections = payload.corrections.model_dump(exclude_none=True) if payload.corrections else None
            return processing.review_invoice(invoice_id, payload.action, payload.reviewer, payload.comment, corrections)
        except processing.NotFound as e:
            raise HTTPException(404, str(e))
        except ReviewError as e:
            raise HTTPException(409 if e.kind == "conflict" else 422, str(e))

    @app.get("/metrics")
    def get_metrics():
        return processing.metrics()

    return app


app = create_app()
