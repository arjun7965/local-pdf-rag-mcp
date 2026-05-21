"""Tests for the chunker.

Synthesizes PDFs with reportlab so we don't have to check binary fixtures
into the repo. The token-budget assertion is the most important one — it
guards the silent regression documented in CLAUDE.md's 'Known gotchas'.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from local_pdf_rag_mcp.chunking import (
    PdfExtractionError,
    chunk_pages,
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


_PARA = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
)


def _long_pages(n_pages: int = 3, paras_per_page: int = 6) -> list[str]:
    return ["\n\n".join([_PARA] * paras_per_page) for _ in range(n_pages)]


@pytest.mark.parametrize("target", [80, 120, 200, 350])
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
