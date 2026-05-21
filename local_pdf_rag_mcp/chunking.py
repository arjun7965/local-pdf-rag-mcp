"""PDF text extraction and chunking.

Extraction is page-aware so every chunk can carry the page number it came
from, which is what makes citations possible. Chunking is paragraph-aware
with token-budgeted packing and overlap, so we avoid cutting sentences in
half while keeping chunks small enough to stay token-cheap at query time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pypdf


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance."""

    text: str
    source: str          # filename the chunk came from
    page: int            # 1-indexed page number
    chunk_index: int     # ordinal of this chunk within the document
    metadata: dict = field(default_factory=dict)


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


def extract_pages(pdf_path: Path) -> list[str]:
    """Return a list of page texts (1 entry per page).

    Raises PdfExtractionError for encrypted PDFs we can't open and for
    image-only / scanned PDFs that yield no extractable text (OCR is out
    of scope for v1).
    """
    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:  # pypdf raises a variety of types
        raise PdfExtractionError(f"Could not open {pdf_path.name}: {exc}") from exc

    if reader.is_encrypted:
        # Try the empty-password path; many "encrypted" PDFs open with it.
        try:
            reader.decrypt("")
        except Exception as exc:
            raise PdfExtractionError(
                f"{pdf_path.name} is encrypted and requires a password."
            ) from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    if not any(p.strip() for p in pages):
        raise PdfExtractionError(
            f"{pdf_path.name} has no extractable text. It may be scanned or "
            f"image-only; OCR is not supported in this version."
        )
    return pages


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) to avoid a tokenizer dependency."""
    return max(1, len(text) // 4)


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_oversized(para: str, target_tokens: int) -> list[str]:
    """Break a paragraph that alone exceeds the target into sentence-packed
    sub-units, so the token budget is actually respected. Falls back to a
    hard character split if a single 'sentence' is still too large (e.g. a
    table row with no sentence punctuation)."""
    if _approx_tokens(para) <= target_tokens:
        return [para]
    pieces: list[str] = []
    buf: list[str] = []
    buf_tok = 0
    for sent in _SENTENCE_SPLIT.split(para):
        stok = _approx_tokens(sent)
        if stok > target_tokens:
            if buf:
                pieces.append(" ".join(buf))
                buf, buf_tok = [], 0
            # Hard split on character budget for pathological cases.
            limit = target_tokens * 4
            for i in range(0, len(sent), limit):
                pieces.append(sent[i : i + limit])
            continue
        if buf_tok + stok > target_tokens and buf:
            pieces.append(" ".join(buf))
            buf, buf_tok = [], 0
        buf.append(sent)
        buf_tok += stok
    if buf:
        pieces.append(" ".join(buf))
    return pieces


def chunk_pages(
    pages: list[str],
    source: str,
    target_tokens: int = 350,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Pack page text into ~target_tokens chunks with overlap.

    Paragraph boundaries are respected where possible. A chunk never spans
    pages so its page number stays unambiguous; this keeps citations honest
    at the cost of slightly smaller chunks near page breaks.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    for page_no, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue

        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(page_text) if p.strip()]
        buffer: list[str] = []
        buffer_tokens = 0

        def flush(buf: list[str]) -> None:
            nonlocal chunk_index
            if not buf:
                return
            text = "\n\n".join(buf).strip()
            if not text:
                return
            chunks.append(
                Chunk(
                    text=text,
                    source=source,
                    page=page_no,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        # Pre-split any paragraph larger than the target so the budget holds.
        units: list[str] = []
        for para in paragraphs:
            units.extend(_split_oversized(para, target_tokens))

        # Cap overlap well below the target so that overlap + one fresh unit
        # cannot blow the budget on the next flush.
        eff_overlap = min(overlap_tokens, target_tokens // 3)

        for para in units:
            ptok = _approx_tokens(para)
            if buffer_tokens + ptok > target_tokens and buffer:
                flush(buffer)
                # Carry a bounded textual tail forward as overlap. Operate on
                # the joined text (not whole units) so a single large unit
                # can't smuggle a full chunk's worth of tokens into the next.
                joined = "\n\n".join(buffer)
                tail = joined[-eff_overlap * 4 :]
                buffer = [tail] if tail.strip() else []
                buffer_tokens = _approx_tokens(tail) if buffer else 0
            buffer.append(para)
            buffer_tokens += ptok

        flush(buffer)

    return chunks
