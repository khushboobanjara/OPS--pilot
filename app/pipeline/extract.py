from app.extraction.regex_extractor import extract_fields
from app.pipeline.base import PipelineContext, Stage

REQUIRED = ["vendor_name", "invoice_number", "total_amount", "due_date"]
RECOMMENDED = ["invoice_date", "currency"]  # po_number is genuinely optional


class ExtractStage(Stage):
    name = "extract"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        fields = extract_fields(ctx.email.get("subject", ""), ctx.email.get("body", ""))

        ctx.invoice = {k: (f.value if f else None) for k, f in fields.items()}
        ctx.invoice["field_confidence"] = {k: (f.confidence if f else 0.0) for k, f in fields.items()}

        for k in REQUIRED:
            if fields[k] is None:
                ctx.flags.append(f"Missing required field: {k}")
        for k in RECOMMENDED:
            if fields[k] is None:
                ctx.flags.append(f"Missing field: {k}")
        for k, f in fields.items():
            if f and f.confidence < 0.7:
                ctx.flags.append(f"Low-confidence {k} ({f.source})")

        scored = REQUIRED + RECOMMENDED
        conf = sum(fields[k].confidence if fields[k] else 0.0 for k in scored) / len(scored)
        ctx.invoice["extraction_confidence"] = round(conf, 4)
        ctx.risk_signals["extraction_uncertainty"] = round(1 - conf, 4)
        return ctx
