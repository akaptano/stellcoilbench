#!/usr/bin/env python3
"""
Chunking policy for PDF text.

Splits long text into overlapping chunks for embedding.
Keeps chunk→page mapping for citations.
"""
from __future__ import annotations

import re
import sys
from typing import Any


def chunk_text(
    text: str,
    *,
    max_chars: int = 512,
    overlap: int = 64,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Split text into overlapping chunks for embedding.

    Splits on paragraph boundaries first, then sentences, then by character.
    Each chunk includes overlap from the previous to preserve context.

    Parameters
    ----------
    text : str
        Text to chunk.
    max_chars : int, optional
        Maximum characters per chunk (default 512).
    overlap : int, optional
        Overlap between consecutive chunks (default 64).
    page : int, optional
        Page number for citation (default 1).

    Returns
    -------
    list[dict]
        [{"text": str, "page": int}, ...]
    """
    if not text.strip():
        return []
    chunks: list[dict[str, Any]] = []
    # Split on paragraphs first, then sentences, then by char
    paragraphs = re.split(r"\n\s*\n", text)
    current = ""
    current_page = page
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append({"text": current, "page": current_page})
            # Start new chunk; try to split para if too long
            while len(para) > max_chars:
                split_at = para[:max_chars].rfind(". ")
                if split_at < max_chars // 2:
                    split_at = max_chars
                chunks.append({"text": para[:split_at + 1].strip(), "page": current_page})
                para = para[split_at + 1:].strip()
                if overlap and para:
                    overlap_text = para[:overlap]
                    para = para[overlap:]
                    current = overlap_text
                else:
                    current = para
                    para = ""
            if para:
                current = para
    if current:
        chunks.append({"text": current, "page": current_page})
    return chunks


def chunk_pages(pages: list[str], chunk_to_page: list[int] | None = None) -> list[dict[str, Any]]:
    """Chunk a list of page texts into overlapping segments.

    Parameters
    ----------
    pages : list[str]
        Text of each page.
    chunk_to_page : list[int] | None, optional
        Page index (1-based) for each page. Defaults to [1, 2, ...].

    Returns
    -------
    list[dict]
        [{"text": str, "page": int}, ...] for all chunks.
    """
    if chunk_to_page is None:
        chunk_to_page = list(range(1, len(pages) + 1))
    all_chunks: list[dict[str, Any]] = []
    for i, page_text in enumerate(pages):
        page_num = chunk_to_page[i] if i < len(chunk_to_page) else i + 1
        all_chunks.extend(chunk_text(page_text, page=page_num))
    return all_chunks


def main() -> int:
    """Read JSON from stdin (from extract_pdf), output chunked JSON."""
    import json
    data = json.load(sys.stdin)
    pages = data.get("pages", data.get("chunks", []))
    chunk_to_page = data.get("chunk_to_page", list(range(1, len(pages) + 1)))
    result = chunk_pages(pages, chunk_to_page)
    print(json.dumps({"chunks": result, "path": data.get("path", "")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
