from app.chunker import chunk_document, count_tokens
from app.parsers.base import Block, BlockType, ParsedDocument

_DOC = {"document_id": "doc-1", "document_title": "Handbook", "source_label": "hr"}


def _parsed(*blocks: Block, file_type: str = "md") -> ParsedDocument:
    return ParsedDocument(file_type=file_type, blocks=list(blocks))


def _para(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text)


def _heading(text: str, level: int) -> Block:
    return Block(type=BlockType.HEADING, text=text, level=level)


def _sentences(text: str) -> list[str]:
    import re

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def test_sizing_and_overlap_respect_token_budget():
    paragraph = " ".join(f"Sentence number {i}." for i in range(12))
    sentence_tokens = count_tokens("Sentence number 5.")
    chunk_size = sentence_tokens * 4
    # Overlap must be able to hold a whole sentence for any overlap to occur (segments
    # are never split), so size the budget to exactly one sentence.
    chunk_overlap = sentence_tokens + 1

    chunks = chunk_document(
        _parsed(_para(paragraph)), **_DOC, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    assert len(chunks) > 1
    # Each chunk stays near the budget (may exceed by at most the last added sentence).
    for chunk in chunks:
        assert count_tokens(chunk.chunk_text) <= chunk_size + sentence_tokens + 1
    # Consecutive chunks overlap by at least one whole sentence.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(_sentences(prev.chunk_text)) & set(_sentences(nxt.chunk_text))


def test_never_breaks_mid_sentence():
    paragraph = " ".join(f"Sentence number {i} here." for i in range(10))
    original = set(_sentences(paragraph))
    chunks = chunk_document(
        _parsed(_para(paragraph)), **_DOC, chunk_size=10, chunk_overlap=3
    )
    for chunk in chunks:
        for sentence in _sentences(chunk.chunk_text):
            assert sentence in original  # every sentence is intact, none partial


def test_giant_paragraph_no_punctuation_stays_whole():
    paragraph = "word " * 200  # no sentence boundaries anywhere
    chunks = chunk_document(
        _parsed(_para(paragraph)), **_DOC, chunk_size=12, chunk_overlap=4
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_text == paragraph.strip()
    assert count_tokens(chunks[0].chunk_text) > 12  # oversized, not truncated


def test_single_oversized_sentence_not_split():
    sentence = "This one sentence is deliberately quite long and keeps going for a while."
    chunks = chunk_document(
        _parsed(_para(sentence)), **_DOC, chunk_size=5, chunk_overlap=1
    )
    assert len(chunks) == 1
    assert chunks[0].chunk_text == sentence


def test_no_headings_enrichment_uses_title_only():
    chunks = chunk_document(_parsed(_para("Body text.")), **_DOC, chunk_size=500)
    chunk = chunks[0]
    assert chunk.section_heading is None
    assert chunk.embed_text == "Handbook\n\nBody text."
    assert "Handbook" not in chunk.chunk_text


def test_nested_headings_use_nearest_preceding_heading():
    chunks = chunk_document(
        _parsed(
            _heading("Top", 1),
            _para("Intro under top."),
            _heading("Mid", 2),
            _para("Mid body."),
            _heading("Deep", 3),
            _para("Deep body."),
        ),
        **_DOC,
        chunk_size=500,
    )
    headings = [c.section_heading for c in chunks]
    assert headings == ["Top", "Mid", "Deep"]
    deep = chunks[-1]
    assert deep.embed_text == "Handbook > Deep\n\nDeep body."


def test_enrichment_and_citation_text_are_separated():
    chunks = chunk_document(
        _parsed(_heading("Policy", 1), _para("Reset your badge.")),
        **_DOC,
        chunk_size=500,
    )
    chunk = chunks[0]
    assert chunk.chunk_text == "Reset your badge."
    assert chunk.embed_text == "Handbook > Policy\n\nReset your badge."
    assert chunk.chunk_text in chunk.embed_text
    assert chunk.embed_text != chunk.chunk_text


def test_chunk_id_is_deterministic_and_formatted():
    parsed = _parsed(_para("Alpha."), _heading("H", 1), _para("Beta beta beta."))
    first = chunk_document(parsed, **_DOC, chunk_size=500)
    second = chunk_document(parsed, **_DOC, chunk_size=500)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    for index, chunk in enumerate(first):
        assert chunk.chunk_id == f"doc-1:{index}"
        assert chunk.chunk_index == index
