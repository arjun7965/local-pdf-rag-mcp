"""PDF text extraction and chunking.

Extraction is page-aware so every chunk can carry the page number it came
from, which is what makes citations possible. Chunking is paragraph-aware
with token-budgeted packing and overlap, so we avoid cutting sentences in
half while keeping chunks small enough to stay token-cheap at query time.

Text extraction uses pdfplumber (MIT, built on pdfminer.six) rather than
pypdf. On real-world PDFs with multi-column layouts, footnotes, justified
text, or unusual character spacing — common in research papers and
specs — pdfplumber preserves word boundaries and reading order much more
reliably, which directly improves what the embedder can match against.
PyMuPDF would be slightly faster but is AGPL-3.0, which would impose
copyleft constraints on anyone consuming this MIT-licensed server.

Table-aware extraction is opt-in (PDF_RAG_TABLES=1). When on, ruled tables
are detected per page, serialized one record per row ("Field: Foo; Bits:
0-3; ..."), and interleaved with prose by vertical position. Each row stays
intact in a single chunk so a query about one row matches that row's record
directly, instead of the cells being scattered by plain text extraction.
Detection is conservative (ruling-line strategy) to avoid mangling prose,
and any page with no detected table falls back to the plain prose path.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance."""

    text: str
    source: str          # filename the chunk came from
    page: int            # 1-indexed page number
    chunk_index: int     # ordinal of this chunk within the document
    chunk_type: str = "prose"   # "prose" or "table"
    metadata: dict = field(default_factory=dict)


@dataclass
class Segment:
    """An ordered piece of a page: either prose text or a table's rows.

    A page is a list of these in reading order. Prose and table segments
    never share a chunk, so a chunk is wholly one or the other.
    """

    kind: str                                  # "prose" or "table"
    prose: str = ""                            # set when kind == "prose"
    rows: list[str] = field(default_factory=list)  # row-records when "table"


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


def tables_enabled() -> bool:
    """Whether table-aware extraction is on. Opt-in via PDF_RAG_TABLES=1."""
    return os.environ.get("PDF_RAG_TABLES", "0") == "1"


def _open_pdf(pdf_path: Path):
    """Open a PDF, mapping failures to PdfExtractionError. pdfplumber tries
    the empty password automatically, so a PDF 'encrypted' with no real
    password just opens."""
    try:
        return pdfplumber.open(str(pdf_path))
    except PDFPasswordIncorrect as exc:
        raise PdfExtractionError(
            f"{pdf_path.name} is encrypted and requires a password."
        ) from exc
    except Exception as exc:
        raise PdfExtractionError(f"Could not open {pdf_path.name}: {exc}") from exc


def _no_text_error(pdf_path: Path) -> PdfExtractionError:
    return PdfExtractionError(
        f"{pdf_path.name} has no extractable text. It may be scanned or "
        f"image-only; OCR is not supported in this version."
    )


# A decorative watermark is rendered far larger than body text. These bound
# the heuristic that strips it before extraction.
_WATERMARK_SIZE_RATIO = 3.0      # a char is watermark if size >= ratio x modal size
_WATERMARK_MAX_FRACTION = 0.15   # ...unless such chars are common, in which case
                                 # the page is genuinely large-text (e.g. a cover)


def _dewatermark(page):
    """Return the page with oversized watermark characters removed.

    Decorative watermarks (e.g. a diagonal 'CapitalFlowsResearch.com' across
    the page) render much larger than body text, and pdfplumber interleaves
    their letters into the extracted text, corrupting chunks. We drop chars
    whose size is a large multiple of the page's modal (body) size. This is a
    no-op when there are no outlier-sized chars, so ordinary PDFs are
    unaffected, and it backs off when oversized text is common so a genuine
    cover/title page isn't gutted. Non-char objects (the ruling lines table
    detection needs) are always kept.
    """
    chars = page.chars
    if not chars:
        return page
    modal = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
    if modal <= 0:
        return page
    threshold = modal * _WATERMARK_SIZE_RATIO
    oversized = sum(1 for c in chars if c.get("size", 0) >= threshold)
    if oversized == 0 or oversized > len(chars) * _WATERMARK_MAX_FRACTION:
        return page
    return page.filter(
        lambda obj: obj.get("object_type") != "char" or obj.get("size", 0) < threshold
    )


def extract_pages(pdf_path: Path) -> list[str]:
    """Return a list of page texts (1 entry per page).

    Raises PdfExtractionError for encrypted PDFs we can't open and for
    image-only / scanned PDFs that yield no extractable text (OCR is out
    of scope for v1).
    """
    pdf = _open_pdf(pdf_path)
    pages: list[str] = []
    try:
        for page in pdf.pages:
            try:
                pages.append(_dewatermark(page).extract_text() or "")
            except Exception:
                pages.append("")
    finally:
        pdf.close()

    if not any(p.strip() for p in pages):
        raise _no_text_error(pdf_path)
    return pages


def _serialize_table(raw_rows: list) -> list[str]:
    """Turn a pdfplumber table (list of rows of cells) into one record per
    data row: "Header1: cell1; Header2: cell2; ...". The first non-empty row
    is treated as the header; blank header cells fall back to colN. Empty
    rows and empty cells are dropped. A single-row table degrades to a plain
    joined line."""
    rows = [
        [(cell or "").replace("\n", " ").strip() for cell in (row or [])]
        for row in (raw_rows or [])
    ]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    if len(rows) == 1:
        return ["; ".join(c for c in rows[0] if c)]

    header = [h if h else f"col{i + 1}" for i, h in enumerate(rows[0])]
    records: list[str] = []
    for row in rows[1:]:
        pairs = []
        for i, cell in enumerate(row):
            if cell:
                col = header[i] if i < len(header) else f"col{i + 1}"
                pairs.append(f"{col}: {cell}")
        if pairs:
            records.append("; ".join(pairs))
    return records


def _segment_page(page) -> list[Segment]:
    """Split one page into prose/table segments in reading order.

    Uses pdfplumber's default (ruling-line) table detection — conservative
    by design, so whitespace-aligned prose isn't misread as a table. Prose
    is pulled from the vertical bands between tables so order is preserved.
    """
    tables = page.find_tables()
    if not tables:
        return [Segment("prose", page.extract_text() or "")]

    tables = sorted(tables, key=lambda t: t.bbox[1])
    segments: list[Segment] = []
    cursor = 0.0
    bottom = float(page.height)

    def add_prose_band(top: float, bot: float) -> None:
        if bot <= top:
            return
        band = page.crop((0, top, page.width, bot))
        text = band.extract_text() or ""
        if text.strip():
            segments.append(Segment("prose", text))

    for table in tables:
        t_top, t_bottom = float(table.bbox[1]), float(table.bbox[3])
        add_prose_band(cursor, t_top)
        rows = _serialize_table(table.extract())
        if rows:
            segments.append(Segment("table", rows=rows))
        cursor = max(cursor, t_bottom)

    add_prose_band(cursor, bottom)
    return segments


def extract_page_segments(pdf_path: Path) -> list[list[Segment]]:
    """Like extract_pages, but returns ordered prose/table segments per page.

    Only used when table-aware extraction is enabled. Any per-page failure
    in table detection falls back to plain prose for that page, so a bad
    table never loses a whole page.
    """
    pdf = _open_pdf(pdf_path)
    pages_segments: list[list[Segment]] = []
    try:
        for page in pdf.pages:
            try:
                clean = _dewatermark(page)
            except Exception:
                clean = page
            try:
                pages_segments.append(_segment_page(clean))
            except Exception:
                try:
                    pages_segments.append([Segment("prose", clean.extract_text() or "")])
                except Exception:
                    pages_segments.append([Segment("prose", "")])
    finally:
        pdf.close()

    has_text = any(
        (s.kind == "prose" and s.prose.strip()) or (s.kind == "table" and s.rows)
        for segs in pages_segments
        for s in segs
    )
    if not has_text:
        raise _no_text_error(pdf_path)
    return pages_segments


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


def _chunk_segmented(
    pages_segments: list[list[Segment]],
    source: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Core packer. Each page is a list of prose/table segments; a chunk
    never spans pages or segment kinds. Prose packs paragraph units with a
    bounded overlap tail; table segments pack whole row-records with no
    overlap (each row is a discrete record). Oversized units hard-split as a
    last resort."""
    chunks: list[Chunk] = []
    chunk_index = 0
    eff_overlap = min(overlap_tokens, target_tokens // 3)

    for page_no, segments in enumerate(pages_segments, start=1):

        def flush(buf: list[str], ctype: str) -> None:
            nonlocal chunk_index
            if not buf:
                return
            sep = "\n\n" if ctype == "prose" else "\n"
            text = sep.join(buf).strip()
            if not text:
                return
            chunks.append(
                Chunk(
                    text=text,
                    source=source,
                    page=page_no,
                    chunk_index=chunk_index,
                    chunk_type=ctype,
                )
            )
            chunk_index += 1

        for seg in segments:
            if seg.kind == "prose":
                paragraphs = [
                    p.strip() for p in _PARAGRAPH_SPLIT.split(seg.prose) if p.strip()
                ]
                units: list[str] = []
                for para in paragraphs:
                    units.extend(_split_oversized(para, target_tokens))

                buffer: list[str] = []
                buffer_tokens = 0
                for para in units:
                    ptok = _approx_tokens(para)
                    if buffer_tokens + ptok > target_tokens and buffer:
                        flush(buffer, "prose")
                        # Carry a bounded textual tail forward as overlap.
                        # Operate on the joined text (not whole units) so a
                        # single large unit can't smuggle a full chunk's
                        # worth of tokens into the next.
                        joined = "\n\n".join(buffer)
                        tail = joined[-eff_overlap * 4 :]
                        buffer = [tail] if tail.strip() else []
                        buffer_tokens = _approx_tokens(tail) if buffer else 0
                    buffer.append(para)
                    buffer_tokens += ptok
                flush(buffer, "prose")
            else:
                # Table rows are discrete records: pack them up to the budget
                # but never split a row (except a pathologically huge one) and
                # never carry overlap between rows.
                units = []
                for row in seg.rows:
                    units.extend(_split_oversized(row, target_tokens))

                buffer = []
                buffer_tokens = 0
                for row in units:
                    rtok = _approx_tokens(row)
                    if buffer_tokens + rtok > target_tokens and buffer:
                        flush(buffer, "table")
                        buffer = []
                        buffer_tokens = 0
                    buffer.append(row)
                    buffer_tokens += rtok
                flush(buffer, "table")

    return chunks


def chunk_pages(
    pages: list[str],
    source: str,
    target_tokens: int = 250,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Pack page text into ~target_tokens chunks with overlap.

    Paragraph boundaries are respected where possible. A chunk never spans
    pages so its page number stays unambiguous; this keeps citations honest
    at the cost of slightly smaller chunks near page breaks.

    The default of 250 sits just under the 256-token max_seq_length of the
    default embedding model (all-MiniLM-L6-v2). If you swap in a model with
    a larger window, raise this to match — otherwise you're paying for
    capacity you don't use.
    """
    pages_segments = [[Segment("prose", text)] for text in pages]
    return _chunk_segmented(pages_segments, source, target_tokens, overlap_tokens)


def chunk_page_segments(
    pages_segments: list[list[Segment]],
    source: str,
    target_tokens: int = 250,
    overlap_tokens: int = 60,
) -> list[Chunk]:
    """Chunk the prose/table segments produced by extract_page_segments.
    Same budgeting as chunk_pages; table chunks are tagged chunk_type="table"."""
    return _chunk_segmented(pages_segments, source, target_tokens, overlap_tokens)
