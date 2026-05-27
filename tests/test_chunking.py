"""Tests for the chunker.

Synthesizes PDFs with reportlab so we don't have to check binary fixtures
into the repo. The token-budget assertion is the most important one — it
guards the silent regression documented in CLAUDE.md's 'Known gotchas'.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

from local_pdf_rag_mcp.chunking import (
    PdfExtractionError,
    chunk_page_segments,
    chunk_pages,
    extract_page_segments,
    extract_pages,
)


def _write_pdf(path: Path, pages: list[str]) -> None:
    c = canvas.Canvas(str(path))
    for page_text in pages:
        if page_text:
            text_obj = c.beginText(50, 750)
            for line in page_text.split("\n"):
                text_obj.textLine(line)
            c.drawText(text_obj)
        c.showPage()
    c.save()


def _write_table_pdf(path: Path, header: list[str], rows: list[list[str]]) -> None:
    """Render a ruled table so pdfplumber's line-based detection finds it."""
    doc = SimpleDocTemplate(str(path))
    table = Table([header] + rows)
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    doc.build([table])


def _write_watermarked_pdf(path: Path, body_lines: list[str], mark: str) -> None:
    """Body text at 11pt plus one oversized (60pt) watermark glyph."""
    c = canvas.Canvas(str(path))
    c.setFont("Helvetica", 11)
    text_obj = c.beginText(72, 700)
    for line in body_lines:
        text_obj.textLine(line)
    c.drawText(text_obj)
    c.setFont("Helvetica", 60)
    c.drawString(250, 400, mark)
    c.showPage()
    c.save()


_PARA = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
)


def _long_pages(n_pages: int = 3, paras_per_page: int = 6) -> list[str]:
    return ["\n\n".join([_PARA] * paras_per_page) for _ in range(n_pages)]


@pytest.mark.parametrize("target", [80, 120, 200, 250, 350])
def test_chunk_pages_respects_token_budget(target: int) -> None:
    chunks = chunk_pages(
        _long_pages(),
        source="synthetic.pdf",
        target_tokens=target,
        overlap_tokens=target // 4,
    )
    assert chunks, "expected at least one chunk"
    sizes = [len(c.text) // 4 for c in chunks]
    # 1.4x slack matches the ~4 chars/token approximation note in CLAUDE.md.
    assert max(sizes) <= target * 1.4, f"over budget at {target}: {max(sizes)}"


def test_chunk_pages_no_empty_chunks() -> None:
    chunks = chunk_pages(_long_pages(), source="synthetic.pdf")
    assert all(c.text.strip() for c in chunks)


def test_chunk_pages_page_numbers_in_range() -> None:
    pages = _long_pages(n_pages=4)
    chunks = chunk_pages(pages, source="synthetic.pdf")
    assert all(1 <= c.page <= len(pages) for c in chunks)


def test_chunk_pages_all_blank_produces_no_chunks() -> None:
    assert chunk_pages(["", "   ", "\n\n"], source="blank.pdf") == []


def test_chunk_pages_tiny_doc_produces_one_chunk() -> None:
    chunks = chunk_pages(["A single short line of text."], source="tiny.pdf")
    assert len(chunks) == 1
    assert chunks[0].page == 1


def test_extract_pages_roundtrip(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, ["First page text.", "Second page text."])
    pages = extract_pages(pdf)
    assert len(pages) == 2
    assert "First" in pages[0]
    assert "Second" in pages[1]


def test_extract_pages_raises_on_no_text(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, ["", ""])
    with pytest.raises(PdfExtractionError):
        extract_pages(pdf)


def test_chunk_pages_tags_chunks_as_prose() -> None:
    chunks = chunk_pages(_long_pages(), source="synthetic.pdf")
    assert chunks
    assert all(c.chunk_type == "prose" for c in chunks)


# --- Table-aware extraction (PDF_RAG_TABLES path) ---


def test_table_extraction_preserves_rows(tmp_path: Path) -> None:
    pdf = tmp_path / "table.pdf"
    _write_table_pdf(
        pdf,
        ["Field", "Bits", "Description"],
        [["Foo", "0-3", "the foo field"], ["Bar", "4-7", "the bar field"]],
    )
    page_segments = extract_page_segments(pdf)
    chunks = chunk_page_segments(page_segments, source="table.pdf")

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert table_chunks, "expected at least one table chunk"

    joined = "\n".join(c.text for c in table_chunks)
    # Each row's cells stay together as a record, keyed by the header.
    assert "Field: Foo" in joined and "Bits: 0-3" in joined
    assert "Field: Bar" in joined and "Bits: 4-7" in joined


def test_table_chunks_respect_token_budget(tmp_path: Path) -> None:
    pdf = tmp_path / "bigtable.pdf"
    rows = [[f"Row{i}", f"{i}-{i + 1}", f"description number {i}"] for i in range(40)]
    _write_table_pdf(pdf, ["Field", "Bits", "Description"], rows)
    page_segments = extract_page_segments(pdf)
    chunks = chunk_page_segments(
        page_segments, source="bigtable.pdf", target_tokens=120, overlap_tokens=30
    )
    assert chunks
    sizes = [len(c.text) // 4 for c in chunks]
    assert max(sizes) <= 120 * 1.4, f"table chunk over budget: {max(sizes)}"


def test_extract_page_segments_raises_on_no_text(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    _write_pdf(pdf, ["", ""])
    with pytest.raises(PdfExtractionError):
        extract_page_segments(pdf)


def test_extract_strips_oversized_watermark(tmp_path: Path) -> None:
    pdf = tmp_path / "wm.pdf"
    _write_watermarked_pdf(
        pdf, ["Body text line one here.", "Body text line two here."], mark="Z"
    )
    text = extract_pages(pdf)[0]
    assert "Body text line one" in text
    assert "Body text line two" in text
    # The 60pt watermark glyph is dropped; body (11pt) survives.
    assert "Z" not in text


def test_extract_keeps_text_when_no_watermark(tmp_path: Path) -> None:
    # No oversized glyphs -> dewatermark is a no-op, body unchanged.
    pdf = tmp_path / "plain.pdf"
    _write_pdf(pdf, ["Ordinary page with normal sized text only."])
    text = extract_pages(pdf)[0]
    assert "Ordinary page with normal sized text only" in text
