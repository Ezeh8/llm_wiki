from app.parsers.base import (
    PARSE_ERROR_MESSAGE,
    Block,
    BlockType,
    FileParseError,
    ParsedDocument,
)
from app.parsers.docx_parser import parse_docx
from app.parsers.markdown import parse_md
from app.parsers.pdf import parse_pdf
from app.parsers.txt import parse_txt

# Extension-keyed registry — the seam for the v2 CSV/Excel/HTML upgrade path (#43):
# register a new parser here, no other changes.
PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "txt": parse_txt,
    "md": parse_md,
}

_ALIASES = {"markdown": "md", "text": "txt"}


def parse_document(file_type: str, data: bytes) -> ParsedDocument:
    key = file_type.lower().lstrip(".")
    key = _ALIASES.get(key, key)
    parser = PARSERS.get(key)
    if parser is None:
        # A genuinely unknown type — the Validator (Step 8) already gates extensions,
        # so this is a programming error, not a client-facing parse failure.
        raise ValueError(f"no parser registered for file type: {file_type!r}")
    return parser(data)


__all__ = [
    "Block",
    "BlockType",
    "ParsedDocument",
    "FileParseError",
    "PARSE_ERROR_MESSAGE",
    "PARSERS",
    "parse_document",
    "parse_pdf",
    "parse_docx",
    "parse_txt",
    "parse_md",
]
