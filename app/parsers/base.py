"""Shared intermediate representation and error type for all file parsers.

Parsers turn raw file bytes into an ordered list of structural `Block`s (headings and
paragraphs). This is deliberately structure-preserving, not a flat blob: the Chunker
(Step 5) splits on headings first, then paragraph boundaries within oversized sections,
then sentences — so it needs the boundaries the parser already knows, not to re-derive
them. Heading `text` doubles as the "section heading" for context enrichment (Node 10).
"""

import re
from enum import Enum

from pydantic import BaseModel

# Client-safe message for any parse failure (mapped to HTTP 422 in Step 12). Never
# leak library internals to the client (D-43).
PARSE_ERROR_MESSAGE = "Could not read this file. It may be corrupted or password-protected."


class FileParseError(Exception):
    def __init__(self, message: str = PARSE_ERROR_MESSAGE) -> None:
        super().__init__(message)


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"


class Block(BaseModel):
    type: BlockType
    text: str
    # Heading depth (1 = top level). None for paragraphs.
    level: int | None = None
    # 1-based source page. Populated for PDF; None for the others.
    page: int | None = None


class ParsedDocument(BaseModel):
    file_type: str  # pdf / docx / txt / md
    blocks: list[Block]


def decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FileParseError() from exc


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
