try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24.3 only exposes the legacy `fitz` name
    import fitz

from app.parsers.base import (
    Block,
    BlockType,
    FileParseError,
    ParsedDocument,
    collapse_whitespace,
)


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise FileParseError() from exc

    try:
        # Encrypted PDFs surface as needs_pass rather than a raised error; we have no
        # password, so treat them as unreadable.
        if doc.needs_pass:
            raise FileParseError()

        blocks: list[Block] = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            # "blocks" gives layout-detected text blocks (paragraph-ish); tuple[6] is the
            # block type (0 = text, 1 = image).
            for raw in page.get_text("blocks"):
                block_type = raw[6] if len(raw) > 6 else 0
                text = collapse_whitespace(raw[4] or "")
                if block_type != 0 or not text:
                    continue
                blocks.append(
                    Block(type=BlockType.PARAGRAPH, text=text, page=page_index + 1)
                )
    except FileParseError:
        raise
    except Exception as exc:
        raise FileParseError() from exc
    finally:
        doc.close()

    return ParsedDocument(file_type="pdf", blocks=blocks)
