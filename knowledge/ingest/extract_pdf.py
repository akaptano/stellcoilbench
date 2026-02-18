#!/usr/bin/env python3
"""
Extract text and page boundaries from PDFs for KB ingestion.

Produces: (pages, text_chunks, chunk_to_page_map)
Requires: pymupdf (pip install pymupdf) or PyPDF2 as fallback.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import fitz  # pymupdf
    _HAS_PYMUPDF = True
except ImportError:
    _HAS_PYMUPDF = False

try:
    from pypdf import PdfReader
    _HAS_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfReader
        _HAS_PYPDF = True
    except ImportError:
        _HAS_PYPDF = False


def extract_pdf(path: Path) -> tuple[list[str], list[str], list[int]]:
    """Extract text from a PDF file for KB ingestion.

    Uses pymupdf (preferred) or pypdf. Produces one chunk per non-empty page
    by default; chunk_pages() can further split long pages.

    Parameters
    ----------
    path : Path
        Path to the PDF file.

    Returns
    -------
    tuple[list[str], list[str], list[int]]
        (pages, chunks, chunk_to_page) where:
        - pages: Full text of each page.
        - chunks: Text chunks (one per page for now).
        - chunk_to_page: 1-based page index for each chunk.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ImportError
        If neither pymupdf nor pypdf is installed.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if _HAS_PYMUPDF:
        doc = fitz.open(path)
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            pages.append(page.get_text())
        doc.close()
    elif _HAS_PYPDF:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
    else:
        raise ImportError("Install pymupdf or pypdf: pip install pymupdf")

    # Simple chunking: one chunk per page
    chunks = [p.strip() for p in pages if p.strip()]
    chunk_to_page = list(range(1, len(chunks) + 1))
    return pages, chunks, chunk_to_page


def main() -> int:
    """Read PDF path, output JSON with pages, chunks, chunk_to_page."""
    if len(sys.argv) < 2:
        print("Usage: extract_pdf.py <path/to/file.pdf>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    pages, chunks, chunk_to_page = extract_pdf(path)
    out = {
        "path": str(path),
        "num_pages": len(pages),
        "pages": pages,
        "chunks": chunks,
        "chunk_to_page": chunk_to_page,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
