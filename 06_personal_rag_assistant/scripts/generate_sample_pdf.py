"""Pure-Python minimal PDF 1.4 generator for creating real pypdf-readable sample PDFs."""

from __future__ import annotations

from pathlib import Path


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_text_pdf(pages: list[list[str]], output_path: str | Path) -> Path:
    """Create a valid multi-page PDF 1.4 document with standard Helvetica Type1 font."""
    objects: list[bytes] = []

    # Object 1: Catalog
    # Object 2: Pages
    # Object 3: Font (Helvetica)
    # Then for each page: Page object + Content stream object
    num_pages = len(pages)
    page_obj_ids = [4 + 2 * i for i in range(num_pages)]

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects.append(f"<< /Type /Pages /Kids [ {kids} ] /Count {num_pages} >>".encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, lines in enumerate(pages):
        page_id = 4 + 2 * i
        content_id = page_id + 1
        stream_cmds = ["BT", "/F1 11 Tf", "50 760 Td", "15 TL"]
        for idx, line in enumerate(lines):
            escaped = _escape_pdf_text(line)
            if idx == 0:
                stream_cmds.append(f"({escaped}) Tj")
            else:
                stream_cmds.append(f"T* ({escaped}) Tj")
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin-1")

        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        content_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )
        objects.append(page_dict)
        objects.append(content_obj)

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    out.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(out))
    return dest


def generate_default_corpus_pdf(output_path: str | Path) -> Path:
    pages = [
        [
            "Python Asyncio Internals - Event Loop Scheduling",
            "Python asyncio executes coroutines cooperatively on a single-threaded event loop.",
            "When a coroutine reaches an await expression on an I/O future, control yields back",
            "to the event loop so other ready tasks can make forward progress without OS threads.",
            "Blocking system calls such as time.sleep stall every concurrent task on the loop.",
            "CPU-bound or synchronous I/O routines should be offloaded via asyncio.to_thread.",
        ],
        [
            "Graceful Shutdown and CancelledError Discipline",
            "During graceful shutdown, pending tasks receive asyncio.CancelledError at their next",
            "await point. Services should catch asyncio.CancelledError only to run cleanup blocks",
            "such as flushing SQLite transactions or closing httpx AsyncClient pools, and then",
            "re-raise CancelledError unless intentionally terminating the supervisor loop.",
            "Structured cancellation prevents zombie background tasks and resource leaks.",
        ],
    ]
    return build_text_pdf(pages, output_path)


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "data" / "corpus" / "05_python_async_internals.pdf"
    created = generate_default_corpus_pdf(target)
    print(f"Generated sample PDF: {created}")
