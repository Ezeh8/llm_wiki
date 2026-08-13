import re

from app.parsers.base import Block, BlockType, ParsedDocument, decode_text

# One or more blank lines separate paragraphs.
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def parse_txt(data: bytes) -> ParsedDocument:
    text = decode_text(data)
    blocks = [
        Block(type=BlockType.PARAGRAPH, text=chunk)
        for chunk in (part.strip() for part in _PARAGRAPH_SPLIT.split(text))
        if chunk
    ]
    return ParsedDocument(file_type="txt", blocks=blocks)
