import re

from app.parsers.base import Block, BlockType, ParsedDocument, decode_text

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_md(data: bytes) -> ParsedDocument:
    text = decode_text(data)
    blocks: list[Block] = []
    paragraph: list[str] = []
    in_fence = False

    def flush() -> None:
        joined = "\n".join(paragraph).strip()
        if joined:
            blocks.append(Block(type=BlockType.PARAGRAPH, text=joined))
        paragraph.clear()

    for line in text.splitlines():
        stripped = line.strip()

        # Track code fences so a leading '#' inside a fence isn't read as a heading.
        if stripped.startswith("```"):
            in_fence = not in_fence
            paragraph.append(line)
            continue

        if not in_fence:
            heading = _ATX_HEADING.match(line)
            if heading:
                flush()
                blocks.append(
                    Block(
                        type=BlockType.HEADING,
                        text=heading.group(2).strip(),
                        level=len(heading.group(1)),
                    )
                )
                continue
            if stripped == "":
                flush()
                continue

        paragraph.append(line)

    flush()
    return ParsedDocument(file_type="md", blocks=blocks)
