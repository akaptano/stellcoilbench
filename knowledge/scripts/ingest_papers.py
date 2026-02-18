#!/usr/bin/env python3
"""
Ingest PDF papers from the papers manifest into the Knowledge Base.

Reads knowledge/papers_manifest.jsonl (produced by fetch_papers.py), extracts
text from each PDF, chunks it into overlapping segments, embeds with
sentence-transformers, and upserts into the Qdrant ``paper_chunks`` collection.

The paper_chunks collection is used by the KB server for semantic search over
the stellarator literature (e.g. when generating research briefs or enriching
the proposer context).

Usage
-----
    cd /path/to/stellcoilbench
    python knowledge/scripts/ingest_papers.py

Dependencies
------------
- sentence-transformers (pip install sentence-transformers)
- qdrant-client (pip install qdrant-client)
- knowledge.ingest.extract_pdf, knowledge.ingest.chunk (pymupdf or pypdf for PDF extraction)

Environment
----------
- KB_QDRANT_URL: Qdrant server URL (default http://localhost:6333)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    """Load manifest, extract and chunk PDFs, embed and upsert to Qdrant.

    Returns 0 on success, 1 on missing dependencies or Qdrant failure.
    Skips papers whose PDF path does not exist.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kb-url",
        type=str,
        default="http://localhost:8000",
        help="KB server URL",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "knowledge" / "papers_manifest.jsonl",
        help="Path to papers_manifest.jsonl",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"No manifest: {args.manifest}", file=sys.stderr)
        return 0

    # Load manifest
    entries = []
    for line in args.manifest.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not entries:
        print("No papers in manifest.", file=sys.stderr)
        return 0

    # Embedder
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers", file=sys.stderr)
        return 1

    # Extract + chunk
    try:
        from knowledge.ingest.extract_pdf import extract_pdf
        from knowledge.ingest.chunk import chunk_pages
    except ImportError:
        print("Ensure knowledge.ingest is importable.", file=sys.stderr)
        return 1

    # Qdrant client (direct; Qdrant runs separately from KB server)
    import os
    qdrant_url = os.environ.get("KB_QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        client = QdrantClient(url=qdrant_url)
    except Exception as e:
        print(f"Qdrant client failed: {e}. Ingest papers via KB API when available.", file=sys.stderr)
        return 1

    # Ensure paper_chunks collection
    try:
        client.get_collection("paper_chunks")
    except Exception:
        client.create_collection(
            "paper_chunks",
            vectors_config={"size": 384, "distance": "Cosine"},
        )

    total = 0
    for entry in entries:
        paper_id = entry.get("id", "unknown")
        path = _REPO_ROOT / entry.get("path", "")
        title = entry.get("title", paper_id)
        if not path.exists():
            print(f"Skip {paper_id}: file not found {path}", file=sys.stderr)
            continue
        try:
            pages, chunks, chunk_to_page = extract_pdf(path)
        except Exception as e:
            print(f"Skip {paper_id}: extract failed {e}", file=sys.stderr)
            continue
        chunked = chunk_pages(pages, chunk_to_page)
        vectors = embedder.encode([c["text"] for c in chunked], convert_to_numpy=True)
        points = []
        for i, (chunk, vec) in enumerate(zip(chunked, vectors)):
            point_id = hashlib.sha256(f"{paper_id}:{i}".encode()).hexdigest()[:16]
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_text": chunk["text"][:2000],
                        "page": chunk.get("page", 1),
                    },
                )
            )
        client.upsert(collection_name="paper_chunks", points=points)
        total += len(points)
        print(f"Ingested {paper_id}: {len(points)} chunks")

    print(f"Total: {total} chunks from {len(entries)} papers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
