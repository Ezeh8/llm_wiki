"""Structure-aware chunking (PRD Node 10, chunking half).

Consumes the Step 4 `ParsedDocument` and produces `Chunk`s. Pure logic, no LangGraph and
no `document_file` dependency — the node wrapper (Step 8) calls this on already-extracted
text and can then set `document_file = None` for free (state cleanup / checkpoint bloat).

Three-tier split priority (PRD): headings/sections first, paragraph boundaries within an
oversized section, sentence boundaries as the last resort. A sentence is never broken —
if a single sentence exceeds `chunk_size` it becomes one oversized chunk rather than
being cut. Overlap is applied by carrying whole trailing segments (never a partial
sentence) up to `chunk_overlap` tokens, and only within a section.
"""

import re
from functools import lru_cache

import tiktoken

from app.domain import Chunk
from app.parsers.base import BlockType, ParsedDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=1)
def _encoder() -> "tiktoken.Encoding":
    # cl100k_base is a general-purpose BPE used only for chunk-sizing (NOT the embedding
    # model's tokenizer). Fetched once on first use — pre-cache in the image (Step 17).
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def _sections(parsed: ParsedDocument) -> list[tuple[str | None, list[str]]]:
    # Group blocks into (nearest-preceding-heading, [paragraph text, ...]) sections.
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    paragraphs: list[str] = []
    for block in parsed.blocks:
        if block.type == BlockType.HEADING:
            if paragraphs or heading is not None:
                sections.append((heading, paragraphs))
            heading = block.text
            paragraphs = []
        else:
            paragraphs.append(block.text)
    if paragraphs or heading is not None:
        sections.append((heading, paragraphs))
    return sections


def _segments(paragraphs: list[str], chunk_size: int) -> list[tuple[str, int]]:
    # Tier 2 -> Tier 3: keep whole paragraphs as segments; only explode a paragraph into
    # sentences when it alone exceeds chunk_size. A monster sentence stays whole.
    segments: list[tuple[str, int]] = []
    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)
        if tokens <= chunk_size:
            segments.append((paragraph, tokens))
        else:
            for sentence in _split_sentences(paragraph):
                segments.append((sentence, count_tokens(sentence)))
    return segments


def _tail_overlap(
    current: list[tuple[str, int]], overlap: int
) -> list[tuple[str, int]]:
    seed: list[tuple[str, int]] = []
    total = 0
    for segment in reversed(current):
        if total + segment[1] > overlap:
            break
        seed.insert(0, segment)
        total += segment[1]
    return seed


def _pack(
    segments: list[tuple[str, int]], chunk_size: int, overlap: int
) -> list[list[str]]:
    chunks: list[list[str]] = []
    seed: list[tuple[str, int]] = []
    i = 0
    while i < len(segments):
        current = list(seed)
        current_tokens = sum(tok for _, tok in current)
        added = 0
        while i < len(segments):
            text, tokens = segments[i]
            # Always take at least one new segment (progress); then stop before overflow.
            if added >= 1 and current_tokens + tokens > chunk_size:
                break
            current.append((text, tokens))
            current_tokens += tokens
            i += 1
            added += 1
            if current_tokens >= chunk_size:
                break
        chunks.append([text for text, _ in current])
        seed = _tail_overlap(current, overlap)
    return chunks


def _enrich(text: str, document_title: str, heading: str | None) -> str:
    header = f"{document_title} > {heading}" if heading else document_title
    return f"{header}\n\n{text}"


def chunk_document(
    parsed: ParsedDocument,
    *,
    document_id: str,
    document_title: str,
    source_label: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for heading, paragraphs in _sections(parsed):
        segments = _segments(paragraphs, chunk_size)
        if not segments:
            continue
        for texts in _pack(segments, chunk_size, chunk_overlap):
            original = " ".join(texts).strip()
            if not original:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}:{index}",
                    chunk_index=index,
                    document_id=document_id,
                    document_title=document_title,
                    source_label=source_label,
                    section_heading=heading,
                    chunk_text=original,
                    embed_text=_enrich(original, document_title, heading),
                )
            )
            index += 1
    return chunks
