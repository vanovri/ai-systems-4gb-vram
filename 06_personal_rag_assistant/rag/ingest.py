"""Document ingestion (.txt, .md, .pdf) and recursive character/sentence chunking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from rag.models import DocumentChunk

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".pdf"})

_DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
)


def normalize_whitespace(text: str) -> str:
    """Normalize line endings and collapse excessive blank lines while preserving paragraphs."""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _recursive_split(
    text: str,
    chunk_size: int,
    separators: tuple[str, ...],
) -> list[str]:
    """Recursively split text using progressively finer separators until segments fit chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0]
    remaining_seps = separators[1:] if len(separators) > 1 else ("",)

    if sep == "":
        return [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size)
            if text[i : i + chunk_size].strip()
        ]

    if sep not in text:
        return _recursive_split(text, chunk_size, remaining_seps)

    raw_parts = text.split(sep)
    pieces: list[str] = []
    for idx, part in enumerate(raw_parts):
        if not part:
            continue
        if sep in {". ", "! ", "? ", "; ", ", "} and idx < len(raw_parts) - 1:
            candidate = part + sep.rstrip()
        else:
            candidate = part
        pieces.append(candidate)

    merged: list[str] = []
    current = ""
    joiner = sep if sep in {"\n\n", "\n", " "} else " "

    for piece in pieces:
        if len(piece) > chunk_size:
            if current.strip():
                merged.append(current.strip())
                current = ""
            merged.extend(_recursive_split(piece, chunk_size, remaining_seps))
            continue

        if not current:
            current = piece
        elif len(current) + len(joiner) + len(piece) <= chunk_size:
            current = f"{current}{joiner}{piece}"
        else:
            merged.append(current.strip())
            current = piece

    if current.strip():
        merged.append(current.strip())

    return merged


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    separators: tuple[str, ...] = _DEFAULT_SEPARATORS,
) -> list[tuple[str, int, int]]:
    """Split text into overlapping chunks returning (chunk_text, start_char, end_char).

    Validates chunk_size > 0 and 0 <= chunk_overlap < chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    base_segments = _recursive_split(cleaned, chunk_size, separators)
    if not base_segments:
        return []

    chunks: list[tuple[str, int, int]] = []
    search_cursor = 0

    for idx, seg in enumerate(base_segments):
        seg_clean = seg.strip()
        if not seg_clean:
            continue

        if idx > 0 and chunk_overlap > 0:
            prev_text = chunks[-1][0]
            max_tail = min(chunk_overlap, max(0, chunk_size - len(seg_clean) - 1))
            if max_tail >= 16:
                raw_tail = prev_text[-max_tail:].strip()
                space_idx = raw_tail.find(" ")
                tail = raw_tail[space_idx + 1 :].strip() if space_idx > 0 else ""
                if tail and not seg_clean.startswith(tail):
                    chunk_body = f"{tail} {seg_clean}".strip()
                else:
                    chunk_body = seg_clean
            else:
                chunk_body = seg_clean
        else:
            chunk_body = seg_clean

        found_at = cleaned.find(seg_clean[:40], max(0, search_cursor - chunk_overlap))
        start_char = found_at if found_at >= 0 else search_cursor
        end_char = min(len(cleaned), start_char + len(seg_clean))
        search_cursor = end_char
        chunks.append((chunk_body, start_char, end_char))

    return chunks


def _sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")
    return safe or "doc"


def read_pdf_pages(file_path: Path) -> list[tuple[int, str]]:
    """Extract (1-based page number, text) tuples from a PDF file using pypdf."""
    reader = PdfReader(str(file_path))
    pages: list[tuple[int, str]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        cleaned = normalize_whitespace(extracted)
        if cleaned:
            pages.append((page_idx, cleaned))
    return pages


def ingest_raw_document(
    text: str,
    source_file: str,
    page: int = 1,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    """Chunk an in-memory text document into DocumentChunk instances."""
    stem = _sanitize_stem(source_file)
    spans = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        DocumentChunk(
            chunk_id=f"{stem}_p{page}_c{idx:03d}",
            text=chunk_body,
            source_file=source_file,
            page=page,
            start_char=start_char,
            end_char=end_char,
        )
        for idx, (chunk_body, start_char, end_char) in enumerate(spans)
    ]


def ingest_file(
    file_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    base_dir: Path | None = None,
) -> list[DocumentChunk]:
    """Ingest a single .txt, .md, or .pdf file into DocumentChunk objects."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if base_dir is not None:
        try:
            source_label = path.resolve().relative_to(base_dir.resolve()).as_posix()
        except ValueError:
            source_label = path.name
    else:
        source_label = path.name

    chunks: list[DocumentChunk] = []
    if ext == ".pdf":
        for page_num, page_text in read_pdf_pages(path):
            chunks.extend(
                ingest_raw_document(
                    text=page_text,
                    source_file=source_label,
                    page=page_num,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )
    else:
        content = path.read_text(encoding="utf-8")
        chunks.extend(
            ingest_raw_document(
                text=content,
                source_file=source_label,
                page=1,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks


def ingest_directory(
    directory: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    recursive: bool = True,
) -> list[DocumentChunk]:
    """Ingest all supported documents (.txt, .md, .pdf) from a directory in deterministic order."""
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    pattern_iter: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    files = sorted(
        p for p in pattern_iter if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    all_chunks: list[DocumentChunk] = []
    for file_path in files:
        all_chunks.extend(
            ingest_file(
                file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                base_dir=root,
            )
        )
    return all_chunks
