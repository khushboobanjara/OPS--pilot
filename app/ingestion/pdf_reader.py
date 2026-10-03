from pathlib import Path

import pdfplumber


def read_pdf_text(path: str | Path) -> str:
    """Extract text from a text-based PDF. Scanned PDFs need OCR (a later upgrade)."""
    with pdfplumber.open(path) as pdf:
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    if not text.strip():
        raise ValueError(f"No extractable text in {path} (scanned PDF? OCR needed)")
    return text
