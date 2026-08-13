# LLM Wiki — Build Log

This file is the implementer's persistent memory across the 20-step build. It records
project layout, conventions, and per-step notes. It is SEPARATE from the PRD
(`llm-wiki-prd.md`), whose Section 7 Decision Ledger is LOCKED. When a convention here
ever conflicts with the PRD, the PRD wins.

---

## Project Conventions (established Step 1)

- **Language / runtime:** Python only. `requires-python >= 3.11` (uses `X | None` unions).
- **Dependency management:** `pyproject.toml` (setuptools backend). NOT requirements.txt.
  Install with `pip install -e .`.
- **App package:** everything lives under `app/`. Sub-packages by concern
  (`app/db/`, and later `app/adapters/`, `app/graph/`, `app/api/`, `app/mcp/`).
- **Config:** `app/config.py` — a single pydantic-settings `Settings` class, cached via
  `get_settings()` (`@lru_cache`). Reads from env + `.env`. `extra="ignore"` so the full
  `.env.example` (seeded with vars for later steps) never breaks startup. Add new fields
  to `Settings` as each step needs them. Env var names are UPPER_SNAKE; fields are
  lower_snake (case-insensitive mapping).
- **ORM style:** SQLAlchemy 2.0 declarative with `Mapped[...]` + `mapped_column(...)`.
  Async only — `asyncpg` driver, `postgresql+asyncpg://` URLs.
- **DB Base:** `app/db/base.py` holds `Base(DeclarativeBase)` with a `MetaData`
  naming-convention (stable pk/fk/uq/ix/ck names) so Alembic diffs and downgrades are
  deterministic.
- **Models:** `app/db/models.py`. UUID PKs use `postgresql.UUID(as_uuid=True)` with
  Python-side `default=uuid.uuid4` (app generates IDs; no DB sequences anywhere).
  Timestamps are `DateTime(timezone=True)` with Python default `datetime.now(timezone.utc)`
  (helper `_utcnow`). No server_defaults — the app owns value generation.
- **Session:** `app/db/session.py` exposes module-level `engine` and
  `async_session_factory` (`async_sessionmaker`, `expire_on_commit=False`), plus
  `get_session()` — an async generator implementing session-per-request with
  rollback-on-exception, ready to become the FastAPI dependency in Step 12.
- **No comments narrating WHAT.** Only WHY-comments for non-obvious PRD constraints.

## File Tree (after Step 1)

```
llm_wiki/
├── llm-wiki-prd.md            # the spec (read-only)
├── BUILD_LOG.md               # this file
├── pyproject.toml
├── .env.example               # all PRD env vars seeded; DB URLs are the live ones for Step 1
├── .gitignore                 # (git NOT initialized — per instructions)
├── alembic.ini                # sqlalchemy.url intentionally blank (loaded from env in env.py)
├── alembic/
│   ├── env.py                 # async migrations; connects as DATABASE_ADMIN_URL
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_schema.py
└── app/
    ├── __init__.py
    ├── config.py
    └── db/
        ├── __init__.py
        ├── base.py
        ├── models.py          # Document, Changelog, AuditLog, CacheEntry, ApiKey
        └── session.py
```

## Two-Role / Grants Approach (Step 1 — key decision)

Requirement: an admin role for migrations (CREATE/ALTER/DROP) and a restricted runtime
role (SELECT/INSERT/UPDATE/DELETE only, no DROP/TRUNCATE).

Chosen model:
- **Admin/migration role** = the identity Alembic connects as via `DATABASE_ADMIN_URL`.
  It OWNS every object it creates, which is what gives it full DDL power. It is
  provisioned once, OUTSIDE migrations — in dev/CI it is simply the Postgres
  `POSTGRES_USER` (superuser); in prod the DBA creates a role with `CREATEROLE` +
  ownership of the target DB. A migration cannot bootstrap the very role that runs it.
- **Restricted runtime role** = created reproducibly *inside* migration `0001`. Its
  username and password are PARSED FROM `DATABASE_URL` (`make_url`), so the entire
  two-role setup is derivable from the two env vars alone — nothing to hand-maintain.
  Created idempotently via a `DO $$ ... IF NOT EXISTS (pg_roles) ... $$` block with
  `LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE`. Password literal is single-quote-escaped;
  role identifier is double-quote-escaped (and rejected if it contains a `"`).
- **Grants** (in the same migration, after tables exist):
  - `GRANT CONNECT` on the DB, `GRANT USAGE` on schema `public`.
  - `GRANT SELECT, INSERT, UPDATE, DELETE` on `documents, changelog, cache, api_keys`.
  - `GRANT SELECT, INSERT ON audit_log` **only** — this enforces the append-only
    contract (no UPDATE/DELETE) at the DB privilege level, the strongest guarantee.
  - TRUNCATE is never granted, and the role never owns tables, so DROP/TRUNCATE both fail.
- **idempotency_key** on `audit_log`: stored as a SINGLE varchar column (the
  concatenation of `thread_id + event_type`) with a UNIQUE constraint. Simpler than a
  composite key and matches the PRD's "store as a single varchar" option. The audit
  writer (Step 9/12) will compose the value.
- **Downgrade** revokes all grants, drops the 5 tables, then `DROP ROLE IF EXISTS`.

Verified end-to-end against a real `postgres:16` container: `alembic upgrade head`
succeeds; connecting as the restricted role, INSERT/SELECT on all tables works while
UPDATE/DELETE on `audit_log`, and TRUNCATE/DROP on `documents`, are all denied;
`downgrade base` drops tables + role; re-`upgrade head` is clean.

### How to run (Step 1)

1. Ensure Postgres is running with an admin role + target DB (dev: `POSTGRES_USER`
   superuser is fine).
2. `cp .env.example .env` and set `DATABASE_ADMIN_URL` (admin) and `DATABASE_URL`
   (the restricted role's desired username/password — the migration creates that role).
3. `pip install -e .`
4. `alembic upgrade head`  → creates 5 tables + the restricted role with correct grants.
5. App runtime (later steps) connects with `DATABASE_URL` and can never DDL.

---

## Step 2 — Adapter-Wall (key decisions)

The graph (Step 8+) NEVER imports qdrant/sqlalchemy clients — only these two adapters
and the Pydantic domain models do. Both adapters: no internal retries (nodes own
retries), validate outputs before returning, raise `AdapterValidationError` on
contract violations.

**New files:** `app/domain.py` (shared Pydantic models), `app/adapters/{__init__,errors,
vector_store,postgres}.py`, `tests/{__init__,test_vector_store_adapter,
test_postgres_adapter}.py`. Added `qdrant-client>=1.9` dep, `[dev]` extra
(pytest/pytest-asyncio), `[tool.pytest.ini_options]` (asyncio_mode=auto). Added
`qdrant_url`/`qdrant_collection` to `Settings`.

**Domain models (`app/domain.py`):** the two PRD state models `RetrievedChunk` &
`Citation`, plus adapter I/O shapes `SparseVector`, `ChunkWithVectors` (store input),
`ScoredChunk` (search output). These are provider-neutral so callers never import a
qdrant/sqlalchemy type.

**Qdrant point/vector schema — Step 7 MUST match this:**
- Named vectors per point: dense **`text`** = 768 dims, **Cosine**
  (Nomic-embed-text-v1.5); sparse **`bm25`** for BM25.
- Point id = deterministic **UUIDv5** of the human `chunk_id`
  (`"{document_id}:{chunk_index}"`), namespace const `_POINT_NAMESPACE` in
  `vector_store.py`. Judgment call: Qdrant point ids must be uint/UUID, so the string
  chunk_id can't be the id; it's kept verbatim in the payload instead.
- Payload keys: `chunk_id, document_id, document_title, source_label, chunk_index,
  chunk_text`. (`source_label` is stored so search can filter by it — D-38/Retriever.)

**VectorStore `search()` returns `list[ScoredChunk]` (judgment call):** it runs TWO
`query_points` calls (dense `using="text"`, sparse `using="bm25"`), each `top_k`, and
merges by `chunk_id` into `ScoredChunk` carrying BOTH `dense_score` and `sparse_score`
(either may be None). Rationale: the Ranker (Step 10) must do the 0.7/0.3 weighted sum
(D-41) + recency tiebreaker, which a single server-fused score could not support. The
Ranker converts `ScoredChunk` → `RetrievedChunk` (the state model) downstream. NOTE for
Step 7/10: no server-side fusion/prefetch is used — fusion is app-side in the Ranker.

**VectorStore adapter surface** (`VectorStoreAdapter(client, collection_name)`):
- `async store_chunks(document_id, chunks_with_vectors: list[ChunkWithVectors]) -> int`
  (validates each dense vec == 768 dims and document_id match; asserts upsert status
  COMPLETED; returns point count). Delete-then-write clean-slate (D-21) is the Storer
  node's job, not the adapter's.
- `async search(query_vector, sparse_vector: SparseVector, top_k, filters=None) -> list[ScoredChunk]`
  (validates query dim; builds Qdrant filter from `source_label`/`document_id`).
- `async delete_by_document_id(document_id) -> None` (idempotent; FilterSelector).
- Factory `build_vector_store()` constructs an `AsyncQdrantClient` from settings.

**Postgres adapter surface** (`PostgresAdapter(session: AsyncSession)`): the adapter
receives a session and **flushes but never commits** — transaction/commit boundary is
owned by the caller's session-per-request lifecycle (Step 12). Methods:
- Documents: `create_document`, `get_document`, `get_document_by_content_hash`
  (for the Duplicate Checker), `list_documents`, `update_document`, `delete_document`.
- Changelog: `create_changelog`, `get_changelog`, `list_changelog`, `update_changelog`
  (supports explicit `set_document_null` for the D-4 SET-NULL path), `delete_changelog`.
- Audit (append-only): `write_audit` — PG `INSERT ... ON CONFLICT (idempotency_key) DO
  NOTHING RETURNING`, falling back to SELECT of the existing row, so replays are
  idempotent and never error (works within the restricted role's INSERT/SELECT-only
  grant). `get_audit`, `list_audit`.
- Cache: `get_cache` (returns None if expired — stale self-heals), `write_cache`
  (upsert on cache_key), `flush_cache` (returns deleted count).
- Auth: `get_key_by_hash` (returns row regardless of `active`; auth layer checks active
  for identical-401), `create_api_key`.
- Pagination: list_* use **keyset** ordering (`<ts> DESC, <id> DESC`) with an opaque
  `cursor` = the last row's id; `_apply_cursor` looks up the anchor row and pages via a
  row-value `tuple_(...) < (...)` comparison. Cursor *encoding* semantics finalize in
  Step 12; the adapter just needs the last id.

**Tests:** `test_vector_store_adapter.py` (8, fully mocked `AsyncMock` Qdrant client —
verifies named/sparse vector construction, dim validation, upsert-status guard, dense+
sparse score merge, filter construction, payload validation, delete selector). All 8
pass. `test_postgres_adapter.py` (6, real PG via `TEST_DATABASE_URL`, skipped if unset —
document CRUD, changelog FK SET NULL on doc delete, audit idempotency, cache
write/read/expiry/flush, key lookup, keyset pagination). All 6 verified green against a
throwaway `postgres:16` container.

## Step 3 — LLM Adapter (key decisions)

Third piece of the adapter-wall: wraps the Generator node's structured-output need
behind one interface, three providers, provider swap = config only (`LLM_PROVIDER` +
`LLM_MODEL_NAME`). Same adapter-wall rules: no internal retries, validate output, fail
with typed errors (the Generator node in Step 10 owns retry + the degraded fallback so
"LLM never blocks retrieval").

**New files:** `app/adapters/llm/{__init__,base,claude,openai_provider,llama}.py`,
`tests/test_llm_adapter.py`. Added `anthropic>=0.40`, `openai>=1.40` deps. Added
`llm_provider/llm_model_name/llm_api_key/llm_temperature/llm_max_tokens/llm_base_url`
to `Settings`; promoted the LLM block in `.env.example` and added `LLM_BASE_URL`.
Added `GenerationResult` to `app/domain.py`.

**Model default stays `claude-sonnet-4-6` (LOCKED by PRD D-45 / env block).** The
claude-api skill defaults to `claude-opus-4-8`, but the PRD explicitly names the model,
so config keeps sonnet-4-6. NOTE for a future me: do not "upgrade" this — it's a locked
decision. (Also: `temperature` is accepted on sonnet-4-6, but would 400 on Opus 4.7+/
Fable 5 — a consideration only if someone later swaps the model.)

**Public interface** — `LLMAdapter.generate(*, system: str, user: str) -> GenerationResult`
(async). The Generator node owns prompt wording and builds `system`/`user` (including
chunk context); the adapter only adds the per-provider structured-output mechanism and
parses. `GenerationResult` (in `app/domain.py`) = `{answer: str, citations: list[Citation]}`
— identical across providers. Errors: `LLMError` base, `LLMGenerationError` (API/transport
failure), `LLMParseError` (couldn't parse/validate structured output). Factory
`build_llm_adapter()` reads settings and returns the right subclass.

**Structured output per provider:**
- **Claude** (`ClaudeAdapter`, `anthropic.AsyncAnthropic`): a **forced tool call** —
  `tools=[{name:"grounded_answer", input_schema: ANSWER_SCHEMA}]` +
  `tool_choice={"type":"tool","name":"grounded_answer"}`. Reads the `tool_use` block's
  `.input`. Judgment call: **no `strict` flag** — strict/structured-outputs support isn't
  guaranteed on sonnet-4-6, and forced tool_use already yields schema-shaped input;
  Pydantic (`GenerationResult.model_validate`) is the validation gate instead.
- **OpenAI** (`OpenAIAdapter`, `openai.AsyncOpenAI`): `chat.completions.create` with
  `response_format={"type":"json_schema","json_schema":{name, schema, strict:True}}`;
  parses `choices[0].message.content` (pure JSON).
- **Llama** (`LlamaAdapter`, `openai.AsyncOpenAI`): **prompt enforcement** — the JSON
  schema is appended to the system prompt, **no `response_format`**; raw text is parsed
  by slicing the outermost `{...}` (tolerates markdown fences/prose) then Pydantic-
  validated. Unparseable → `LLMParseError`.

**Llama transport judgment call (documented for consistency):** the Llama path reuses the
**OpenAI SDK pointed at an OpenAI-compatible endpoint** via `LLM_BASE_URL` (vLLM / Ollama
`/v1` / TGI — the standard real-world Llama serving shape), so no extra HTTP client/dep.
What distinguishes it from the `openai` provider is the structured-output *method*: the
`openai` provider uses native `response_format` json_schema; the `llama` provider uses
prompt enforcement + robust text parsing (per PRD "prompt enforcement for Llama"). The
`openai` client requires a non-empty api_key, so the llama factory passes `"not-needed"`
when `LLM_API_KEY` is blank.

**Shared parsing** (`base.py`): `ANSWER_SCHEMA` (the one target schema, `additionalProperties:
false`, all fields required), `parse_generation(dict|str)` → validates into
`GenerationResult` or raises `LLMParseError`; `_extract_json_object` slices the outermost
brace pair for the prompt-enforced path.

**Import hygiene:** the `llm` subpackage is NOT re-exported from `app/adapters/__init__.py`,
so importing the Postgres/VectorStore adapters does not pull in `anthropic`/`openai`.

**Tests (8, fully mocked `AsyncMock` clients, no network):** Claude forced-tool-call
construction + parse, missing tool block → parse error, API failure → generation error,
invalid tool input → parse error; OpenAI response_format construction + parse, malformed
content → parse error; Llama no-response_format + schema-in-system + fenced-JSON
extraction, unparseable → parse error. All green; factory verified for all three providers.

## Step 4 — File Parsers (key decisions)

Parsing half of Node 10 (Chunker). Four dedicated parsers behind an extension-keyed
**registry**, each returning one shared intermediate representation. Chunking itself is
Step 5 and consumes this IR directly. Parsers receive raw `bytes` (already
extension/size-validated by the Validator, Step 8).

**New files:** `app/parsers/{__init__,base,pdf,docx_parser,txt,markdown}.py`,
`tests/test_parsers.py`. Added `PyMuPDF>=1.24`, `python-docx>=1.1` deps.

**Intermediate representation (`app/parsers/base.py`) — Step 5 builds directly on this:**
```python
class BlockType(str, Enum): HEADING = "heading"; PARAGRAPH = "paragraph"
class Block(BaseModel):
    type: BlockType
    text: str
    level: int | None = None   # heading depth, 1 = top level; None for paragraphs
    page: int | None = None    # 1-based page, PDF only; None otherwise
class ParsedDocument(BaseModel):
    file_type: str             # pdf / docx / txt / md
    blocks: list[Block]        # ordered
```
Rationale for Step 5: `blocks` is an ordered stream where HEADING blocks (with `level`)
delimit sections and PARAGRAPH blocks are the split units. So the chunker can: split on
headings first (section = heading + following paragraphs up to the next heading of equal/
higher level), then split oversized sections on paragraph boundaries (one `Block` each),
then sentences within a paragraph as last resort — without re-parsing structure. A
heading's `text` is the "section heading" to prepend for context enrichment; PDF `page`
is available if page-level citation is wanted later.

**PDF library choice — PyMuPDF (`pymupdf`, legacy `fitz`).** Faster and more robust text
extraction than pdfplumber, and it exposes `doc.needs_pass` so password-protected PDFs
are detected cleanly and mapped to the parse error. Import is future-proofed:
`try: import pymupdf as fitz / except ImportError: import fitz` (PyMuPDF 1.28 deprecates
the `fitz` name).

**One sentence per parser (how each extracts structure):**
- **PDF** (`pdf.py`): `page.get_text("blocks")` yields layout-detected text blocks →
  one PARAGRAPH `Block` per text block, whitespace-collapsed, tagged with 1-based `page`;
  `needs_pass` → parse error.
- **DOCX** (`docx_parser.py`): iterates `document.paragraphs`, mapping paragraph style
  names (`Heading N`, `Title`) to HEADING blocks with `level`, everything else to
  PARAGRAPH; empty paragraphs skipped.
- **TXT** (`txt.py`): UTF-8 decode, split on blank lines (`\n\s*\n`) into PARAGRAPH
  blocks (paragraph detection); no headings.
- **Markdown** (`markdown.py`): line scan — ATX `#{1,6}` markers → HEADING blocks with
  `level`; blank lines separate PARAGRAPH blocks; tracks ``` fences so a `#` inside a code
  block is not misread as a heading.

**Error-handling pattern:** a single typed `FileParseError(message=PARSE_ERROR_MESSAGE)`
with the exact client-safe string `"Could not read this file. It may be corrupted or
password-protected."`. Each parser wraps its library calls in try/except and re-raises as
`FileParseError` — no library internals leak. Step 12 maps this to HTTP 422. Decode
failures (invalid UTF-8) for TXT/MD, and open/parse failures for PDF/DOCX, all funnel to
the same typed error. Naming note: the DOCX module is `docx_parser.py` (not `docx.py`) to
avoid shadowing the top-level `docx` package — same hygiene choice as Step 3's
`openai_provider.py`.

**Registry** (`app/parsers/__init__.py`): `PARSERS = {"pdf","docx","txt","md"}` +
`parse_document(file_type, data)` dispatch (normalizes case/leading dot, aliases
`markdown`→`md`, `text`→`txt`). This is the exact seam for the v2 CSV/Excel/HTML upgrade
(#43) — register a parser, nothing else changes. Unknown type → `ValueError` (a
programming error, since the Validator gates extensions), not a client-facing parse error.

**Tests (11):** valid TXT/MD/DOCX/PDF fixtures generated in-memory (PyMuPDF + python-docx)
verifying structure extraction (paragraph detection, heading levels, fenced-`#` handling,
PDF page numbers); registry dispatch + alias; unknown-type ValueError; a corrupted-file
case per parser and a password-protected PDF, all asserting the exact `FileParseError`
message. All green.

## Step 5 — Chunker (key decisions)

Chunking half of Node 10. Pure function `chunk_document(...)` in `app/chunker.py`,
consumes the Step 4 `ParsedDocument`, emits `list[Chunk]`. No LangGraph, no
`document_file` dependency — it runs on already-extracted text so Step 8's node wrapper
can set `document_file = None` right after (state cleanup / checkpoint bloat) for free.

**New files:** `app/chunker.py`, `tests/test_chunker.py`. Added `tiktoken>=0.7` dep;
`chunk_size`/`chunk_overlap` (500/50) to `Settings`; `Chunk` model to `app/domain.py`.

**Chunk output shape (`app/domain.py` `Chunk`) — Steps 6 & 9 build on this:**
```python
class Chunk(BaseModel):
    chunk_id: str            # f"{document_id}:{chunk_index}" (deterministic, Node 12)
    chunk_index: int         # global 0..N-1 across the whole document
    document_id: str
    document_title: str
    source_label: str        # carried through for Qdrant payload + filtering
    section_heading: str | None
    chunk_text: str          # ORIGINAL text — citations + Qdrant payload
    embed_text: str          # ENRICHED text — what Step 6 embeds
```
Maps cleanly onto Step 2: Storer builds `ChunkWithVectors(chunk_id, chunk_index,
chunk_text=chunk_text, document_id, document_title, source_label, dense/sparse from
embed_text)`; `chunk_id` → Qdrant point via Step 2's uuid5. State field `chunks:
list[dict]` = `[c.model_dump() for c in chunks]` (Step 8 dumps).

**Signature:** `chunk_document(parsed, *, document_id, document_title, source_label,
chunk_size=500, chunk_overlap=50) -> list[Chunk]`. Defaults match `Settings`
CHUNK_SIZE/CHUNK_OVERLAP so Step 8 wires RunnableConfig values straight through.

**Tokenizer choice — tiktoken `cl100k_base`.** Used ONLY for chunk-sizing token counts,
not as the embedding model's tokenizer (Nomic is not tiktoken-based) — fast, well-
supported, and close enough for budgeting. `_encoder()` is `lru_cache`d and lazy;
cl100k_base is fetched on first use, so **pre-cache it in the Docker image (Step 17)** or
set `TIKTOKEN_CACHE_DIR` to avoid a runtime network fetch.

**3-tier split priority (implementation):**
1. **Sections first** — `_sections()` groups blocks into `(nearest-preceding-heading,
   [paragraphs])`. Each section is chunked independently; sections are NOT merged (keeps
   one heading per chunk for clean enrichment), and overlap never crosses a section.
2. **Paragraphs within an oversized section** — `_segments()` keeps each paragraph as one
   segment; only a paragraph that alone exceeds `chunk_size` is exploded further.
3. **Sentences as last resort** — oversized paragraphs split on `(?<=[.!?])\s+`. A single
   sentence longer than `chunk_size` stays whole → one oversized chunk (hard constraint:
   never break mid-sentence).
   `_pack()` greedily fills a chunk with whole segments until adding the next would exceed
   `chunk_size` (always taking ≥1 new segment for progress). **Overlap** = `_tail_overlap()`
   carries whole trailing segments (never a partial sentence) up to `chunk_overlap` tokens
   into the next chunk; if the last segment alone exceeds the overlap budget, the seed is
   empty (correctly no overlap rather than a broken sentence).

**Enrichment vs citation text separation:** `chunk_text` = the original joined segment
text (what citations/`Citation.chunk_text` and the Qdrant payload show). `embed_text` =
`_enrich()` = `f"{title} > {heading}\n\n{chunk_text}"` (or `f"{title}\n\n{chunk_text}"`
when no heading) — this is what Step 6 embeds. Two separate fields, never conflated.

**Tests (8):** token-budget sizing + one-sentence overlap (params derived from real token
counts); never-break-mid-sentence (every chunk sentence matches an original exactly);
adversarial giant no-punctuation paragraph stays whole; single oversized sentence not
split; no-headings → title-only enrichment; nested headings → nearest-preceding heading;
enrichment/citation text separation; chunk_id determinism + `doc:{index}` format. Green.

## Step 6 — Embedding (key decisions)

Nomic embedding for both paths (Node 3 query embed + Node 11 batch embed). Plain async
`Embedder` class in `app/embedding.py`; Step 8 wraps its methods as nodes.

**New files:** `app/embedding.py`, `tests/test_embedding.py`. Added
`sentence-transformers>=3.0` + `einops>=0.7` deps (torch comes transitively; nomic-bert
needs einops). Added `embedding_model_name` (`nomic-embed-text-v1.5`) +
`embedding_batch_size` (50) to `Settings`.

**Model loading (verification-flagged decision):** IN-PROCESS via sentence-transformers
`SentenceTransformer(repo, trust_remote_code=True)`, repo id `nomic-ai/nomic-embed-text-v1.5`
(the loader prefixes `nomic-ai/` when the configured name has no `/`). Confirms Crash
Risk #1/#4: a ~500MB-1GB CPU model held in RAM, not an API. Loading is **lazy + cached**:
`_load_model()` is imported lazily (importing `app.embedding` does NOT pull torch), and
`@lru_cache get_embedder()` is the process-wide singleton whose weights load on the first
embed call. `model.encode(..., normalize_embeddings=True)` → unit vectors (matches Qdrant
Cosine from Step 2). **Nomic task prefixes are applied here** (separate from the Chunker's
title/heading enrichment already in `Chunk.embed_text`): `search_query: ` for queries,
`search_document: ` for chunks — required for Nomic retrieval quality.

**Could NOT test against the real model** (honest note): torch + sentence-transformers +
~500MB-1GB of Nomic weights are too heavy to download in this sandbox. The model is an
**injectable boundary** (`Embedder(model=...)`, `EmbeddingModel` Protocol with `.encode`),
so tests use a `FakeModel` and never touch the network. The real load path (`_load_model`)
is present and correct but exercised only in a real deployment (pre-cache weights in the
Step 17 image; ~8GB VPS per Crash Risk #1).

**Function signatures + return shapes (Steps 7/9/10 depend on these):**
- `async Embedder.embed_query(query_text: str) -> list[float]` — single 768-dim vector
  (query path, `search_query:` prefixed). → state `query_embedding`.
- `async Embedder.embed_chunks(chunks: list[Chunk]) -> list[list[float]]` — one 768-dim
  vector per chunk, **aligned to input order**, embedding each `chunk.embed_text`
  (`search_document:` prefixed). → state `embeddings: list[list[float]]` (aligns with
  `chunks[i]`; Step 9 zips with chunks, Step 7/Storer builds `ChunkWithVectors`).
- `get_embedder() -> Embedder` — cached singleton (lazy real load).
- Errors: `EmbeddingError` (base, raised after exhausted retries) and
  `EmbeddingDimensionError` (count or dim ≠ 768).
- Module constant `EMBEDDING_DIM = 768` (must match `vector_store.EMBEDDING_DIM`).

**Retry / backoff / validation:** retry lives in THIS layer (`retries=1` default → 2
attempts per batch), per PRD Node 11's "1 retry per batch"; built so Step 8 can keep it
here or move it into the node wrapper — I kept it here. **Backoff for rate limits is
effectively N/A for local inference** (no remote rate limit); a configurable
`backoff_seconds` (default 0.0) is retained so an API-hosted embedding fallback could
reuse the same retry path — the retry otherwise guards transient CPU/thread errors.
Batching is `embedding_batch_size` (~50) via `range(0, n, batch)`; **768-dim + count
validation runs on every batch** before results are extended (adapter-wall "validate
outputs" convention). `encode` is offloaded with `asyncio.to_thread` so the blocking
CPU call doesn't stall the event loop.

**Tests (9, model boundary injected):** query prefix + 768 dim; chunk prefix + order
preserved across batches (marker-encoded); batch splitting `[50,50,20]`; empty input →
no model call; wrong dimension → `EmbeddingDimensionError`; count mismatch →
`EmbeddingDimensionError`; retry recovers after one transient failure (2 attempts);
exhausted retries → `EmbeddingError`; cached singleton identity. All green in an isolated
minimal venv (no torch download).

## Step 7 — Qdrant Setup + BM25 Sparse Encoding (key decisions)

Collection provisioning + the shared BM25 sparse-vector encoder. Honors the Step 2
committed schema exactly (names/dim imported from `vector_store` so they can't drift).

**New files:** `app/qdrant_setup.py`, `tests/test_qdrant_setup.py`. Added `fastembed>=0.3`
dep; `sparse_model_name` (`Qdrant/bm25`) to `Settings`. BM25 encoder added to
`app/embedding.py` (co-located with the dense Embedder — all text→vector encoders in one
module). `SparseVector` (Step 2) UNCHANGED — already matches.

**Confirmed Qdrant schema (verified against a real Qdrant container):**
- dense `text`: `VectorParams(size=768, distance=Cosine)`
- sparse `bm25`: `SparseVectorParams(modifier=Modifier.IDF)`
- Provisioning read back from a live collection: `{'text': (768,'Cosine')}`,
  `{'bm25': 'idf'}` — exact match.

**Engineer Verification Task #2 (Qdrant sparse API) — DONE in code.** Findings:
- Current `qdrant-client` (1.19) collection API: `create_collection(collection_name,
  vectors_config={name: VectorParams}, sparse_vectors_config={name: SparseVectorParams})`;
  `collection_exists()` and `delete_collection()` exist on both sync and `AsyncQdrantClient`.
- qdrant-client does NOT compute BM25 — it only stores/searches sparse vectors. The
  BM25 sparse vector must be computed client-side.
- **BM25 approach: fastembed `SparseTextEmbedding("Qdrant/bm25")`** (the Qdrant-endorsed,
  local, no-server-compute encoder). Verified real output: documents (`.embed`) get
  term-frequency values (e.g. 1.674); queries (`.query_embed`) get unit values (1.0).
  The IDF component is applied **server-side** by the collection's `Modifier.IDF`, using
  corpus stats — so documents store TF and Qdrant multiplies IDF at query time. This is
  why the sparse vector config carries `modifier=IDF`. `SparseEmbedding.indices` (int64)
  / `.values` (float64) map cleanly to `SparseVector(indices: list[int], values:
  list[float])` — Step 2's model was already correct; no adjustment needed.
- **End-to-end seam check (real Qdrant + real BM25):** stored 2 chunks via the Step 2
  adapter (named dense+sparse), hybrid-searched "reset my badge" → the BM25 side returned
  a non-zero IDF-weighted `sparse_score` (2.321) on the matching chunk. Steps 2 + 6 + 7
  interoperate.

**Provisioning function:** `async ensure_collection(client: AsyncQdrantClient | None =
None, *, recreate: bool = False) -> bool`. Idempotent: `collection_exists` → returns
False if present, creates + returns True if absent; `recreate=True` drops then rebuilds.
Builds its own `AsyncQdrantClient` from `QDRANT_URL` when none is passed and closes it.
CLI: `python -m app.qdrant_setup` (also the Step 17/README setup step / app-startup hook).

**Shared BM25 encoder home = `app/embedding.py`** (`BM25Encoder`, `get_bm25_encoder()`
lazy singleton — mirrors the dense `Embedder`/`get_embedder`):
- `BM25Encoder.encode_chunks(chunks) -> list[SparseVector]` — over `chunk.embed_text`
  (enriched, same text the dense side embeds), aligned to input order. Step 9 Storer pairs
  these with the dense vectors to build `ChunkWithVectors`.
- `BM25Encoder.encode_documents(texts) -> list[SparseVector]` — low-level.
- `BM25Encoder.encode_query(query_text) -> SparseVector` — uses `.query_embed` (raw user
  query, NO Nomic prefix). Step 10 Retriever passes the result to `adapter.search`.
Model is an injectable boundary (`SparseModel` Protocol) like the dense one.

**Real-instance testing:** YES for both — provisioning verified against a throwaway
`qdrant/qdrant` container (created + idempotent + schema read-back), and BM25 verified
with the real fastembed model (the guarded integration test actually ran, not skipped).
Unit tests also cover the mocked-client and injected-model paths.

**Tests (7):** collection create with correct dense (768/Cosine) + sparse (IDF) config
(mocked client); idempotent-exists (no create); recreate (delete then create); BM25
`encode_chunks` alignment + uses embed_text (injected model); `encode_query` uses the
query path; empty→empty; a guarded REAL fastembed BM25 test asserting valid int indices /
positive float values for doc and query. All green.

## Step 8 — LangGraph State Schema + Graph Wiring (key decisions)

**Built directly in this session (not delegated) after the Step 8 subagent hit the
account's monthly Anthropic spend limit mid-task and terminated cleanly before writing
any graph code** — see the human/assistant exchange in this session for context. All of
Steps 1-7 (subagent-built) were re-read in full before starting, to stay consistent.

**New files:**
```
app/graph/
├── __init__.py
├── state.py            # GraphState (19 fields), Configurable, build_run_config()
├── checkpointer.py      # AsyncPostgresSaver setup + runtime accessor
├── routing.py           # 5 routing functions (4 PRD points + 1 gap-fill)
├── build.py              # build_graph() / compile_graph()
└── nodes/
    ├── __init__.py
    ├── _common.py        # call_with_retry() shared node-layer retry helper
    ├── validator.py       # Node 1 — REAL
    ├── query_path.py      # Nodes 2-8 — cache_checker/embedder_query/retriever REAL;
    │                       # ranker/quality_gate/generator/cache_writer STUB (Step 10)
    ├── ingest_path.py     # Nodes 9-12 — duplicate_checker/chunker_node/
    │                       # embedding_batcher REAL; storer STUB (Step 9)
    ├── delete_path.py     # Node 13 — deleter STUB (Step 11)
    └── audit.py            # Node 14 — REAL
tests/test_graph.py
```
Added `langgraph>=1.0`, `langgraph-checkpoint-postgres>=3.0`, `psycopg[binary]>=3.1`
deps. Added `max_file_size_mb` (20) / `max_query_length` (2000) to `Settings` (already
seeded in `.env.example` as `MAX_FILE_SIZE_MB`/`MAX_QUERY_LENGTH`, unused until now).

**Node wiring status (8 REAL, 6 STUB):**
| Node | Status | Built on | Filled for real in |
|---|---|---|---|
| 1 Validator | REAL | — (self-contained checks) | — |
| 2 Cache Checker | REAL | Step 2 PostgresAdapter | — |
| 3 Embedder (query) | REAL | Step 6 Embedder | — |
| 4 Retriever | REAL | Step 2 VectorStoreAdapter + Step 7 BM25Encoder | — |
| 5 Ranker | STUB (placeholder passthrough) | — | Step 10 |
| 6 Quality Gate | STUB (always "sufficient") | — | Step 10 |
| 7 Generator | STUB (empty answer) | — | Step 10 |
| 8 Cache Writer | STUB (no-op) | — | Step 10 / 13 |
| 9 Duplicate Checker | REAL | Step 2 PostgresAdapter | — |
| 10 Chunker | REAL | Step 4 parsers + Step 5 chunker | — |
| 11 Embedding Batcher | REAL | Step 6 Embedder | — |
| 12 Storer | STUB (no-op) | — | Step 9 |
| 13 Deleter | STUB (no-op) | — | Step 11 |
| 14 Audit Writer | REAL | Step 2 PostgresAdapter | — |

**AsyncPostgresSaver verification (Engineer Verification Task #1) — DONE, against real
langgraph-checkpoint-postgres 3.1.1 + a live Postgres 16 container:**
- Correct import: `from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver`.
- It is an **async context manager**, not a plain constructor:
  `async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:`.
- Requires the **`psycopg`** (v3) driver with the `binary` extra — a different driver
  than the app's `asyncpg`/SQLAlchemy stack — so its connection string must be plain
  `postgresql://`, not `postgresql+asyncpg://`. `to_psycopg_conn_string()` converts.
- `checkpointer.setup()` creates 4 tables: `checkpoint_migrations`, `checkpoints`,
  `checkpoint_blobs`, `checkpoint_writes` — via `CREATE TABLE IF NOT EXISTS` (idempotent).
- **Least-privilege consequence (a real friction point, resolved):** `.setup()` needs
  CREATE TABLE, which the Step 1 restricted runtime role does NOT have. So
  `setup_checkpointer_schema()` (admin bootstrap, idempotent, call at app startup) runs
  `.setup()` via `DATABASE_ADMIN_URL`, then GRANTs SELECT/INSERT/UPDATE/DELETE on the 4
  checkpoint tables to the restricted role (parsed from `DATABASE_URL`, same pattern as
  migration `0001`). Ongoing runtime graph invocations use `build_checkpointer()`,
  connected as the **restricted** role — verified this actually works (not just that
  the connection opens): a full graph run with this checkpointer both wrote a checkpoint
  row and committed an audit_log row using ONLY the restricted role's grants.
- `StateGraph(GraphState)` + `add_node`/`add_conditional_edges`/`compile(checkpointer=)`
  all confirmed against the installed API (1.2.10) via a live toy graph before building
  the real one — no API drift from what the PRD assumes.
- RunnableConfig: `from langchain_core.runnables import RunnableConfig`; nodes receive
  `config["configurable"]` as a plain dict, confirmed via a live toy node.

**Two gap-fill routing branches (Step 8 judgment calls, NOT locked-decision changes):**
the PRD's Section 3 "Routing (4 decision points)" literally lists only 4, but two node
behaviors it already specifies have nowhere to route without a 5th/6th edge:
1. `route_after_validator`: a validation failure (`status == "error"`) routes straight
   to Audit Writer instead of continuing by `operation_type` — otherwise a rejected
   upload would silently continue toward the Chunker.
2. `route_after_duplicate_checker` (new function): a duplicate (`status == "error"`)
   routes straight to Audit Writer instead of Chunker — otherwise Node 9's rejection
   would have no effect and the duplicate would still be chunked/embedded/stored.

**`document_metadata` reuse (judgment call, D-11 stays intact — 19 fields, no new
top-level field added):** the state schema is locked at exactly 19 fields, and several
node behaviors the PRD specifies need MORE scratch data than the 19 fields provide
(content_hash, file_size_bytes, chunk_count, a query's optional filter, file_type for
routing to the right parser). Rather than reopen D-11, all of these live inside
`document_metadata` (typed as a bare `dict` in the PRD table, not a fixed sub-schema):
- Duplicate Checker adds `content_hash`, `file_size_bytes`.
- Chunker adds `chunk_count`.
- **FastAPI (Step 12) is expected to set `document_metadata["file_type"]`** from the
  uploaded filename before invoking the graph — Validator and Chunker both need it and
  no other state field carries it. Flagged clearly for Step 12.
- A query's optional `filter` (`source_label`/`document_id`) lives at
  `document_metadata["filter"]` — the same "generic per-request scratch dict" reuse,
  despite the field's PRD-example name suggesting ingest-only use.

**`retrieved_chunks` type nuance (judgment call, consistent with Step 2's own
BUILD_LOG note):** the PRD table types it `list[RetrievedChunk]`, but between Retriever
and Ranker it actually holds `ScoredChunk`-shaped dicts (dense_score + sparse_score kept
separate) — RetrievedChunk's single `score` field can't carry what the Ranker needs to
compute the 0.7/0.3 weighted sum (D-41). The Step 8 Ranker STUB does a placeholder
reduction (score = dense_score or sparse_score, no real weighting) purely so the DAG is
runnable end-to-end for wiring tests; Step 10 replaces it with the real weighted sum +
recency tiebreaker (D-5), at which point `ranked_chunks` is genuinely
`list[RetrievedChunk]`-shaped as the PRD documents.

**`document_id` generation for a new ingest:** Chunker generates a fresh `uuid4()` when
`state.get("document_id")` is falsy, since chunk_id = document_id + chunk_index (Node
12) needs it before chunking. **Left unresolved for Step 11:** the update path needs to
track BOTH the new document_id (generated here) AND the old one (to delete after the
new one is confirmed stored, per the fortified order, D-2) — state has only one
`document_id` field, so Step 11 will need a mechanism (most likely stashing the old id
in `document_metadata`, consistent with the reuse pattern above). Flagged, not solved.

**Session-per-node Postgres pattern:** RunnableConfig only carries the connection
string/config (the sealed envelope, D-12), never a live session — and LangGraph nodes
are independently resumable units, not a single request-scoped unit like a FastAPI
handler. So every Postgres-touching node (`cache_checker`, `duplicate_checker`,
`audit_writer`) opens its own short-lived session via `app.db.session.async_session_factory`,
uses `PostgresAdapter`, and commits itself if it performed a write (Cache Checker is
read-only, so it just lets the session close; Duplicate Checker's lookup is read-only
too; Audit Writer commits explicitly).

**Node-layer retries** (`app/graph/nodes/_common.call_with_retry`): Duplicate Checker 1,
Retriever 1, Audit Writer 3 all wrap their adapter calls with this per PRD's per-node
retry counts (adapters themselves never retry, D-7). Embedder query's 1 retry is
already inside `Embedder.embed_query` (Step 6), so no double-wrapping. Cache
Checker/Writer are 0-retry/best-effort per PRD, so neither uses this helper.

**Real verification, not just import-level confidence** (throwaway Postgres 16
container, same pattern as Steps 1/2/7): `alembic upgrade head` clean; checkpointer
`setup_checkpointer_schema()` ran for real and is idempotent (2nd call no-ops
correctly); a full `graph.ainvoke()` on the **delete** path, using the real
restricted-role checkpointer, both wrote a checkpoint row AND committed a real
`audit_log` row — confirmed by querying the container directly; a full
`graph.ainvoke()` on the **query cache-hit** path correctly short-circuited to Audit
Writer without touching the embedder. (Did NOT exercise the ingest path's
`embedding_batcher` end-to-end — it would trigger a real ~500MB-1GB Nomic weight
download, same constraint noted in Step 6; unit-level coverage of that logic already
exists in `test_embedding.py`.)

**Tests (25 in `test_graph.py`, all green):** state schema field-set match (19/19);
graph compiles with all 14 nodes; all 5 routing functions (8 cases); Validator (9 cases
— query empty/too-long/valid, ingest missing-fields/bad-extension/valid/oversized,
delete missing-id/valid); conn-string conversion. Plus 3 integration tests gated on
`RUN_GRAPH_INTEGRATION_TESTS=1` (skipped by default; requires
`DATABASE_URL`/`DATABASE_ADMIN_URL` to point at a real Postgres for the whole test
process — see the test file's module docstring for why: Step 1's module-level
`engine`/`async_session_factory` singleton can't be repointed per-test via monkeypatch,
worth revisiting once FastAPI's lifespan exists in Step 12) — checkpointer setup +
idempotency + restricted-role grants; delete-path end-to-end; query cache-hit
end-to-end. Full project suite: **82 passed**.

## Step 9 — Ingestion Path / Storer (key decisions)

Built directly in this session (Sonnet, per explicit instruction — Step 8's Opus
subagent had hit the account spend limit; user chose to continue without switching
back unless a later step needs deeper reasoning). Fills in the Storer stub from Step 8
(`app/graph/nodes/ingest_path.py`) with the real Node 12 logic; the rest of the
ingestion path (Duplicate Checker, Chunker, Embedding Batcher) was already real as of
Step 8.

**Found and fixed a real bug from Step 8** before starting: `embedding_batcher` called
`get_embedder()` without importing it — a `NameError` waiting to happen, never caught
because no Step 8 test exercised that function. Fixed the import; flagging here since
it shipped in a prior "done" step.

**New/changed files:**
```
app/graph/nodes/ingest_path.py   # storer: now REAL (was STUB)
app/adapters/postgres.py          # + upsert_document, set/get_health_flag(s)
app/db/models.py                  # + HealthFlag
alembic/versions/0002_health_flags.py
tests/test_postgres_adapter.py    # + upsert_document, health flag tests
tests/test_graph.py               # + 2 real ingest-path integration tests
```

**Storer implementation** (`app/graph/nodes/ingest_path.py::storer`), exact PRD order,
1 retry on the whole 3-step sequence (each step is independently idempotent, so
retrying from the top is safe):
1. `store.delete_by_document_id(document_id)` — clean slate (D-21), prevents duplicate
   chunks if this node is retried or re-run.
2. `adapter.upsert_document(...)` — Postgres BEFORE Qdrant, so step 3's failure has
   something to roll back.
3. `store.store_chunks(...)` (dense+sparse via `BM25Encoder.encode_chunks`, built from
   `chunk.embed_text` — same text the dense side embeds, consistent with Step 7). If
   this raises, **roll back**: delete the Postgres row just written, re-raise (#24).
Cache flush is a separate, unconditional side effect (`_flush_cache_side_effect`) that
runs regardless of whether the write succeeded — 1 retry; failure never fails the
ingest, it only sets a health flag (see below).

**New Postgres primitive — `upsert_document`:** Storer needs a genuine upsert
("write document metadata to Postgres (upsert by document_id)", Node 12), not
`create_document`'s plain INSERT (which would raise on any document_id conflict,
including a legitimate Storer retry). Added via `ON CONFLICT (document_id) DO UPDATE`,
same style as Step 2's `write_cache`.

**Real bug found and fixed while testing `upsert_document`: SQLAlchemy ORM identity-map
staleness on ORM-enabled `INSERT ... RETURNING`.** Upserting the SAME primary key twice
in one session returned the STALE first-inserted object (unchanged attributes) instead
of the freshly updated row — confirmed via a reproduction script (`first is second ==
True`). Root cause: `Session.execute()` on an ORM-mapped RETURNING statement doesn't
refresh an object already present in the session's identity map unless told to. Fixed
with `execution_options={"populate_existing": True}` on both `upsert_document` (new)
**and Step 2's `write_cache`** (same latent bug, same fix — a repeated cache_key upsert
in one session had the identical risk, just never exercised by Step 2's tests since
each cache_key was only written once per test).

**`health_flags` table (new, migration `0002`) — storage mechanism for D-16/D-24's
"set health flag cache_stale_risk=true":** the PRD doesn't name a storage mechanism for
this (unlike the dead-letter system, which explicitly calls for a persistent volume).
Since everything else in this app is Postgres-backed, a small table
(`name` PK, `value` bool, `updated_at`) is the consistent choice over introducing a
file-based mechanism for just this one flag. Restricted role granted
SELECT/INSERT/UPDATE (no DELETE — flags are cleared via `value=false`, never row
deletion). `PostgresAdapter.set_health_flag/get_health_flag/get_health_flags` — the
last one is for Step 15's health endpoint to read all flags in one call.
Judgment call: a subsequent SUCCESSFUL flush clears a previously-set
`cache_stale_risk=true` flag back to `false` (self-heals) — the PRD only says "set" it
on failure and doesn't mention clearing, but leaving it permanently true after one
transient blip would make the health endpoint reflect history instead of current
state, which seems clearly not the intent.

**Real verification** (throwaway Postgres 16 + Qdrant containers, `alembic upgrade
head` through both `0001`+`0002`, real checkpointer setup): full `graph.ainvoke()` on
the **ingest** path — Validator → Duplicate Checker → Chunker → Embedding Batcher →
Storer → Audit Writer — with a real `.md` file, monkeypatched `Embedder` (fake model,
same reasoning as Step 6/8: avoids a ~500MB-1GB Nomic download, everything else is
real). Confirmed: 2 real Qdrant points written and readable back by `document_id`
filter; Postgres `documents` row correct (title/chunk_count/content_hash); Storer's
cache-flush side effect correctly set `cache_stale_risk=False` (cache was already
empty, flush succeeded); `document_file` correctly cleared to `None` in the final
state. Then re-ingested the IDENTICAL content under a new thread — correctly rejected
as `"Duplicate document"`, and **`document_file` was NOT cleared** (proof
`route_after_duplicate_checker`'s gap-fill routing, added in Step 8, actually skips the
Chunker on a duplicate, not just in unit tests). Also verified the rollback path with a
mocked Qdrant failure: `store_chunks` raising mid-write correctly leaves NO Postgres
document row behind (deleted by the rollback) — no ghost documents (#24).

**Known test-isolation caveat (flagged for Step 18, not a product bug):**
`test_postgres_adapter.py`'s fixture does a raw `Base.metadata.drop_all`/`create_all`
against `TEST_DATABASE_URL` — if that happens to point at the SAME database as
`DATABASE_URL`/`DATABASE_ADMIN_URL` (as it did in this session's verification), it
silently strips the restricted role's grants (created via Alembic's GRANT statements,
which raw metadata create_all doesn't know about), breaking any test that runs after
it in the same process. Not a code defect — purely a shared-fixture-database
collision in ad-hoc verification. Ran `test_postgres_adapter.py` in its own process to
avoid it here; Step 18's proper test harness should give each test module (or the
whole suite) an isolated database so this can't happen by accident.

**Tests:** 2 new `test_postgres_adapter.py` cases (upsert-then-update, health flag
set/get/default-false) — real PG, all green. 2 new `test_graph.py` integration cases
(full ingest end-to-end + duplicate rejection; Qdrant-failure rollback) — gated on
`RUN_GRAPH_INTEGRATION_TESTS=1`, the ingest one additionally guarded/skips if no live
Qdrant collection is reachable (same pattern as Step 7's guarded real-BM25 test). Full
suite (run as two processes per the isolation caveat above): **78 + 8 = 86 passed.**

## Step 10 — Query Path (Ranker, Quality Gate, Generator, Cache Writer) (key decisions)

Built directly in this session (Sonnet, continuing from Step 9's precedent). Fills in
the remaining 4 query-path stubs from Step 8 (`app/graph/nodes/query_path.py`); Cache
Checker/Embedder/Retriever were already real as of Step 8.

**Changed/new files:**
```
app/graph/nodes/query_path.py    # ranker/quality_gate/generator/cache_writer: STUB → REAL
app/domain.py                     # ChunkWithVectors/ScoredChunk + ingested_at
app/adapters/vector_store.py      # payload/_to_scored_chunk carry ingested_at
app/graph/nodes/ingest_path.py    # Storer stamps + threads ingested_at
app/adapters/postgres.py          # upsert_document(ingested_at=...)
tests/test_vector_store_adapter.py, tests/test_graph.py   # new coverage
```

**Ranker recency tiebreak — a genuine cross-step design decision, not just Step 10
logic:** Node 5 needs to "prefer newer documents on score tie," but neither
`ScoredChunk` nor `RetrievedChunk` carried a timestamp, and the PRD explicitly labels
Ranker "pure logic, no external calls" — ruling out a live Postgres lookup on the hot
query path. Fix: `ingested_at` (ISO-8601 UTC string) now travels in the Qdrant payload,
set once by Storer and read directly off every search hit. Required touching 3 already
-"done" files (`domain.py`'s `ChunkWithVectors`/`ScoredChunk`, `vector_store.py`'s
payload write/read, and Step 9's Storer) — a deliberate, scoped extension, not a
redesign. Storer generates `ingested_at` ONCE per write attempt (not per retry) and
passes the identical value to both the Postgres row (`upsert_document`, new
`ingested_at` param — included in both the INSERT and the ON CONFLICT UPDATE branch, so
a Storer-internal retry keeps a consistent timestamp) and every chunk's Qdrant payload,
so the two stores never disagree about a document's age.

**Ranker** (`app/graph/nodes/query_path.py::ranker`): `combined = dense_weight *
dense_score + sparse_weight * sparse_score` (0.7/0.3 default, D-41, both configurable);
sorted by `(combined_score, ingested_at)` descending — Python tuple comparison gives
score-primary/recency-secondary ordering for free, since ISO-8601 UTC strings sort
lexicographically in chronological order.

**Quality Gate** (`quality_gate`): `best_score >= quality_threshold` (0.55 default,
D-38) → `status="sufficient"`, else `"insufficient"`. Pure logic, no external calls —
genuinely true this time, no data needed beyond what Ranker already produced.

**Generator** (`generator`) — the biggest piece, implementing D-39's 3-layer grounding:
- Builds the system/user prompt itself (the LLM adapter, Step 3, only owns per-provider
  structured-output mechanics) enforcing all 5 PRD prompt constraints: answer only from
  chunks, no training-data knowledge, cite chunk_id per claim, state what's missing
  rather than guess, prefer the newer source on conflict.
- Layer 1: the adapter's own schema validation (Step 3).
- Layer 2: strips any citation whose `chunk_id` isn't in `ranked_chunks` — catches a
  model citing something it wasn't given.
- Layer 3: if ALL citations get stripped (zero survive — including the case where the
  LLM returned none at all, which is intentionally treated the same as "all invalid":
  an uncited answer is exactly what D-39 means by unverified), OR the LLM/parse call
  fails outright (`LLMError` — covers both `LLMGenerationError` and `LLMParseError` in
  one except clause), degrade to the PRD's exact fallback string, `citations=[]`,
  `status="degraded"`. 0 retries, and no exception ever escapes this node — "LLM never
  blocks retrieval."
- **`degraded`/`source_chunks` are NOT new state fields:** `degraded` reuses the
  `status` enum's already-documented `"degraded"` value (PRD state table literally
  lists it as a valid `status`); `source_chunks` (the API response field, Step 12) is
  just `ranked_chunks` surfaced conditionally when `status == "degraded"` — the PRD's
  own Node 7 spec says "populate source_chunks with raw ranked_chunks," and
  `ranked_chunks` already holds exactly that. No state schema change needed (D-11
  stays locked at 19 fields); flagged clearly for Step 12 to build the response that way.

**Cache Writer** (`cache_writer`): writes only on `status == "success"`, never on
`"degraded"` — a judgment call, since the PRD doesn't say either way. Caching a
degraded fallback would keep serving "couldn't generate a verified answer" to every
identical question for the full 48h TTL instead of retrying generation, which seemed
clearly worse than not caching it. (`"insufficient"` never reaches this node at all —
Step 8's routing skips straight to Audit Writer.) 0 retries, best-effort, matches
Step 2's `write_cache` (which also got a `populate_existing` fix here — see below).

**Real verification** (throwaway Postgres 16 + Qdrant, full schema/checkpointer/
collection provisioned): ingested a real document (fake dense-embedding model only,
same reasoning as Steps 6/8/9 — avoids a ~500MB-1GB download), then ran a REAL query
through Cache Checker → Embedder → Retriever → Ranker → Quality Gate → Generator (LLM
mocked, since no real Anthropic key in this sandbox — everything else genuinely real:
hybrid Qdrant search, weighted ranking, threshold gate, citation-vs-ranked_chunks
verification) → Cache Writer → Audit Writer. Confirmed: real combined hybrid score
(0.87) passed the threshold; the mocked LLM's citation against the REAL chunk_id it was
given survived verification; cache write succeeded; a SECOND identical query correctly
returned `status="cache_hit"` with the same answer, short-circuiting the whole
pipeline. Separately verified in isolation: quality gate's threshold both ways; the
recency tiebreak (older doc loses to newer doc on an exact score tie); the
citation-stripping degraded path (invalid chunk_id → degraded); the LLM-failure
degraded path (adapter raises → degraded, nothing escapes the node).

**Tests:** 13 new pure-unit tests (quality gate ×3, ranker ×2, generator ×3, cache
writer ×1 — all in `test_graph.py`; `test_vector_store_adapter.py` updated for the new
required `ingested_at` field) + 1 new real end-to-end integration test (ingest → real
hybrid query → verified generation → cache write → cache-hit replay), gated on
`RUN_GRAPH_INTEGRATION_TESTS=1` and guarded to skip if Qdrant is unreachable, same
pattern as Step 9. Full suite (two processes, per the Step 9 test-isolation note):
**88 + 8 = 96 passed.**

## Step 11 — Delete/Update Path (key decisions, including 2 real bugs found + fixed)

Built directly in this session (Sonnet). Fills in the last stub node (Deleter) and
resolves the update path's old/new document_id split that Steps 9-10 explicitly left
open. This step's live end-to-end testing caught two genuine bugs that no unit test
had surfaced — documented in full below since they're the most important part of this
step's work, not just the new Deleter code.

**Changed/new files:**
```
app/graph/nodes/delete_path.py     # deleter: STUB → REAL
app/graph/nodes/ingest_path.py     # chunker_node: resolves old/new document_id split
                                    #   + now self-sufficient for content_hash (bug fix)
app/graph/nodes/_common.py         # + flush_cache_side_effect (moved from ingest_path,
                                    #   now shared by Storer AND Deleter)
app/graph/routing.py               # route_after_storer: bug fix (see below)
tests/test_graph.py                # regression tests for both bugs + new coverage
```

**Old/new document_id split (the update-path design Steps 9/10 flagged as
unresolved):** FastAPI's `PUT /documents/{id}` (Step 12) will seed the initial state's
`document_id` with the OLD document being replaced. D-2's fortified order ("ingest new
FIRST, verify success, THEN delete old — duplicate temporarily better than a hole")
only means anything if the new content lands as a genuinely SEPARATE row — overwriting
the old row in place would leave nothing to fall back to if the write failed partway.
So `chunker_node` (the first node update and ingest share) now: if
`operation_type == "update"` and an incoming `document_id` is present, stashes it as
`document_metadata["previous_document_id"]` (same scratch-dict pattern as
content_hash/chunk_count) and generates a FRESH id for the new document. Deleter reads
`previous_document_id` — not `state.document_id`, which by the time Deleter runs has
already been overwritten with the NEW id — to know what to clean up.

**Deleter** (`app/graph/nodes/delete_path.py`): Qdrant chunks first, then Postgres
metadata — naturally idempotent either way, so no rollback symmetry with Storer is
needed. 1 retry (the PRD only states a retry count for Deleter's cache-flush side
effect, "same as Storer," not the delete itself — applied 1 retry to the delete too,
for consistency with every other DB/Qdrant-touching node in this build; a judgment
call, documented). Changelog FK SET NULL (D-4) needed no application code — it's a
Postgres `ON DELETE SET NULL` foreign key from migration `0001`, so deleting the
`documents` row does it automatically. `flush_cache_side_effect` was pulled out of
`ingest_path.py` into `_common.py` since Deleter needed the exact same side effect
Storer already had.

---

**Bug 1 — found by actually running the update path (no unit test caught this):**
`content_hash`/`file_size_bytes` were ONLY ever computed by Duplicate Checker. That's
fine for ingest (Duplicate Checker always runs first), but the update path's routing
skips Duplicate Checker entirely — updates aren't duplicate-checked, since
intentionally replacing a document isn't a duplicate. Result: every single update hit
a bare `KeyError: 'content_hash'` inside Storer. Fix: `chunker_node` now always
computes both fields itself from the raw bytes it already decodes (idempotent, cheap,
and removes the implicit "Duplicate Checker must have run first" coupling entirely —
Duplicate Checker still independently computes its own copy for its own lookup, which
is necessary since it must decide before chunking even starts).

**Bug 2 — found while fixing Bug 1, and worse:** `route_after_storer` (Step 8) routed
to Deleter for `operation_type == "update"` regardless of whether Storer actually
succeeded. Combined with Bug 1, this meant: Storer fails (KeyError) → routes to
Deleter anyway → Deleter successfully deletes the OLD document → Deleter's
`status="success"` then overwrites Storer's `status="error"` in the final state (all
19 fields overwrite, per the locked state schema) → the whole operation reports
`"success"` while the update actually destroyed the old document and stored nothing.
This is precisely the "hole" D-2's fortified order exists to prevent, produced by the
routing itself rather than a missing safeguard. Fix: `route_after_storer` now also
checks `status != "error"` before routing to Deleter; a failed Storer routes straight
to Audit Writer, leaving the old document untouched. Verified with a live negative
test: mocked `store_chunks` to fail mid-update — the final state correctly reports
`status="error"`, and the old document is confirmed still present in Postgres
afterward.

Both bugs only surfaced because the update path was exercised end-to-end against a
real Postgres+Qdrant, not from any unit test or code review of the individual nodes in
isolation — the failure mode lived entirely in how two already-"working" pieces
(Chunker's metadata contract, Storer's routing) interacted across the ingest/update
path boundary.

---

**Real verification** (throwaway Postgres 16 + Qdrant, full schema/checkpointer/
collection provisioned): ingested a document, attached a changelog entry to it, ran a
real update (new document_id generated, differs from old), confirmed: old document row
gone from Postgres, new row present with correct title, changelog's `document_id`
correctly `NULL` (FK SET NULL verified live, not just by migration inspection), old
document's Qdrant points gone, new document's Qdrant points present. Then ran a genuine
delete on a separate document plus an idempotent re-delete (no error on deleting an
already-gone document). Then the negative test described above (Bug 2's regression
check) with a real mocked Qdrant failure mid-update.

**Tests:** 2 routing unit tests (including a regression test asserting a failed update
does NOT route to Deleter) + 3 Deleter target-resolution unit tests (genuine delete
targets `state.document_id`; update cleanup targets `previous_document_id`, not the new
id; no-previous-id case targets nothing) + 2 real end-to-end integration tests (full
update + FK SET NULL + delete + idempotent re-delete; the Bug 2 regression test with a
simulated Qdrant failure), gated on `RUN_GRAPH_INTEGRATION_TESTS=1`, guarded to skip if
Qdrant is unreachable — same pattern as Steps 9/10. Full suite (two processes, per the
established test-isolation note): **94 + 8 = 102 passed.**

## Step 12 — FastAPI Endpoints (key decisions)

Built directly in this session (Sonnet). The biggest step yet: all 14 routes, auth,
rate limiting, tiered timeouts, error handling, pagination, and wiring HTTP requests
into the compiled LangGraph from Steps 8-11.

**New files:**
```
app/api/
├── app.py                  # create_app(): lifespan (checkpointer+graph), middleware,
│                             # routers, exception handlers. CORS deliberately absent.
├── schemas.py               # all request/response Pydantic models (API-2, locked)
├── errors.py                 # AppError hierarchy + {status,error,message,retryable} body
├── error_mapping.py           # graph state.error (string) -> AppError subclass
├── auth.py                    # API key hash + require_tier() dependency
├── middleware.py               # RateLimitMiddleware, RequestLoggingMiddleware
├── pagination.py                # clamp_limit / paginate (keyset peek-ahead)
├── timeouts.py                   # with_timeout() + the 4 tiered budgets
├── graph_runner.py                 # run_graph(): thread_id + RunnableConfig + ainvoke
└── routers/{documents,query,changelog,audit,health}.py
app/create_api_key.py         # CLI: python -m app.create_api_key --tier admin --actor "..."
tests/test_api.py, tests/test_api_units.py
```
Added `fastapi`, `uvicorn[standard]`, `python-multipart` deps (+ `httpx` to `[dev]`
for `TestClient`). Added `rate_limit_admin/service/employee` to `Settings` (values
already seeded in `.env.example`).

**Admin key creation is a CLI, not an endpoint** (PRD leaves this "TBD by
implementer"): an HTTP endpoint would need an admin key to authenticate the request
that creates the first admin key — a bootstrapping deadlock a CLI run directly against
the database sidesteps entirely. `python -m app.create_api_key --tier admin --actor
"Jane"` prints the raw key once (only its hash is ever stored).

**Auth implemented as a FastAPI dependency (`require_tier(...)`), not raw ASGI
middleware**, even though the PRD lists "Auth" as step 2 of 4 in "Middleware Order."
Tier-per-route naturally fits FastAPI's per-route dependency injection; reimplementing
route-pattern matching inside middleware would be strictly worse for no benefit. The
ordering intent is still satisfied: dependencies resolve before the endpoint body
runs, and `RequestLoggingMiddleware` wraps the whole call chain so it always logs the
real outcome (including 401s) after the fact. Verified LIVE, not just reasoned about
— see below.

**Rate limiter does its own lightweight key lookup, shared with Auth via
`request.state.api_key`:** the PRD wants rate limits bucketed by TIER (30/10/10 per
minute) but also wants the rate limiter to run BEFORE auth extracts/validates the key
— a real tension, since you can't bucket by tier without knowing the tier. Resolved by
having `RateLimitMiddleware` do its own cheap key→tier lookup (stashed on
`request.state` so the `require_tier` dependency downstream reuses it instead of
querying twice) and bucketing by key_hash; a missing/garbage key still gets bucketed
(at the lowest configured tier, as a safe default) rather than skipped, so
unauthenticated floods can't bypass rate limiting by omitting a key. In-memory
sliding-window counters (a `deque` per key_hash) — no Redis, appropriate for
single-process ICL tooling (D-29 defers multi-tenancy/scale-out to v2 anyway).

**Middleware registration order verified LIVE, not just reasoned about:** Starlette
applies the LAST-added middleware OUTERMOST, so `app.py` adds
`RequestLoggingMiddleware` first, `RateLimitMiddleware` second — yielding RateLimit
(outer) → Logging (wraps auth+endpoint) → Auth dependency → endpoint. Proved with two
live tests: (1) flooding `/documents` with **no API key at all** past the limit
produces 401×10 then 429×N — i.e. rate limiting engages even for requests that will
later fail auth, confirming it's genuinely outermost; (2) all three 401 causes (no
key, wrong tier, revoked key) produce byte-identical response bodies.

**A real HTTP-semantics gap, caught by testing, not by reading the PRD twice:**
`DELETE /documents/{id}` and `DELETE /changelog/{id}` are locked to return **204 with
a JSON body** (`{deleted: true, ...}`). FastAPI's normal `status_code=204 +
response_model=...` combination silently DROPS the response body — 204/304 are
treated as body-less by convention, and FastAPI enforces that. Verified this exact
failure mode live (a first draft's DELETE returned an empty body), then fixed both
endpoints to construct the response manually via `JSONResponse(status_code=204,
content=jsonable_encoder(...))`, which bypasses FastAPI's auto-suppression. Re-verified
the body actually arrives over real HTTP.

**Error mapping from graph state to HTTP status** (`error_mapping.py`): the 19-field
state schema (D-11, locked) has no structured error-code field — just a plain `error`
string set by whichever node rejected the operation (Validator, Duplicate Checker,
Chunker's parse-failure passthrough, Storer/Deleter). Rather than reopen the schema,
`error_to_app_error()` matches on the small, FIXED set of literal messages those nodes
already emit (safe because they're code-defined constants, not arbitrary text — the
one case that interpolates an exception's `str()`, Storer/Deleter failures, still has
a fixed, matchable PREFIX).

**Two judgment calls on DELETE semantics, kept deliberately different by design:**
- `DELETE /documents/{id}` returns 204 (success) even for an already-gone document —
  matches Deleter's explicit PRD-stated idempotency (Node 13) and standard idempotent-
  DELETE REST semantics.
- `DELETE /changelog/{id}` returns 404 for a missing entry — changelog has no
  dedicated graph node or idempotency guarantee in the PRD; it's plain CRUD, so
  standard REST 404-on-missing applies instead.
- `PUT /documents/{id}` (update) explicitly checks the target document exists first
  and 404s if not — matches the PRD's 404 table entry ("Document... ID doesn't exist")
  and conventional PUT semantics (replacing something that should already exist),
  even though the graph's own Deleter cleanup half would otherwise silently no-op on a
  nonexistent old id.

**Quality Gate's "insufficient" status has no PRD-given response text** (only the
Generator's "degraded" fallback has an exact locked string) — `POST /query` returns
200 with a plain, clearly-worded placeholder answer, `citations=[]`,
`source_chunks=None`, `degraded=False` (kept distinct from "degraded" per the state
schema's own distinct enum values). A judgment call, documented for Step 16 (MCP) to
map consistently with the `insufficient_results` MCP error code (Section 4).

**Tiered timeouts** (`timeouts.py`, `asyncio.wait_for` wrapping the graph
invocation/DB call per route): query 30s, ingestion 120s, delete/update 30s,
changelog/audit/listing 10s — caught one easy-to-conflate case while wiring
`PUT /documents/{id}`: it shares the `_ingest_or_update` helper with `POST
/documents`, which made it trivial to accidentally give update the 120s ingestion
budget instead of the 30s delete/update budget; fixed by branching on
`operation_type` inside the shared helper. Timeout → 503 (a judgment call — the PRD's
status table doesn't name a dedicated timeout code; 503's "Qdrant or Postgres
unreachable" framing extends naturally to "or too slow").

**Multipart contract judgment call:** the PRD's `POST /documents` request shape says
"multipart — file + metadata {title, source_label, changelog_id?}" without specifying
whether `metadata` is a JSON sub-object or separate form fields. Implemented as
separate `Form(...)` fields (`title`, `source_label`, `changelog_id`) — the standard
FastAPI multipart pattern, and avoids requiring API clients to JSON-encode a
sub-object inside a multipart request.

**`file_type` derivation:** taken from the uploaded filename's extension
(lowercased), fulfilling the contract Step 8/9 already documented as owed to FastAPI
("document_metadata[\"file_type\"] must be set before invoking the graph").

**A real cross-event-loop testing pitfall, found and fixed (not a product bug):**
`TestClient` runs the ASGI app in its own thread with its own event loop (anyio's
blocking portal) per test, but `app.db.session.engine`'s connection pool is a
module-level singleton shared across the whole pytest process. A connection pooled
during one test's (now-dead) loop being handed to a later test's different loop is a
genuine asyncpg "attached to a different loop" `RuntimeError`. Fixed by disposing the
engine's pool (`await engine.dispose()`) at the start of every `api_client` fixture
instantiation, forcing fresh connections against whichever loop is actually live. Also
found and fixed a second, related issue: any test code that called
`async_session_factory()` DIRECTLY (bypassing the running TestClient's own loop, e.g.
to seed API keys or revoke one) hit the same conflict from the opposite direction —
fixed by giving such test-only DB access its own disposable, throwaway engine
(`_run_with_fresh_session`) instead of touching the shared one at all. Both fixes are
purely test-infrastructure concerns; the actual application code was correct
throughout — this is exactly the kind of thing Step 18's dedicated test harness should
formalize once and for all.

**Real verification** (throwaway Postgres 16 + Qdrant, full schema/checkpointer/
collection provisioned, fake dense-embedder + mocked LLM to avoid a Nomic download and
a real Anthropic call): exercised the ENTIRE HTTP surface for real — all 14 routes,
every tier's auth (including wrong-tier and revoked-key rejection with byte-identical
401 bodies), rate limiting (per-tier enforcement, health's exemption, and the
before-auth ordering proof), cursor pagination (disjoint pages, silent capping at
100), the CLI key-creation tool, duplicate/unsupported-extension/oversized-file/empty-
question error paths mapped to the correct status codes, full ingest→list→detail→
update→delete→re-delete lifecycle (with the update correctly producing a new
document_id and 404ing the old one), changelog CRUD, and audit listing/detail
reflecting real recorded events.

**Tests:** 6 pure-unit tests (`test_api_units.py`: pagination clamp/peek-ahead, error
mapping) + 23 real end-to-end HTTP tests (`test_api.py`, gated on
`RUN_GRAPH_INTEGRATION_TESTS=1` like the rest of the graph integration suite) covering
auth (4), health (1), document lifecycle (4), query (3), changelog CRUD (1), audit
(1), rate limiting (3), pagination (1), plus supporting fixtures. Full suite (two
processes, per the established `test_postgres_adapter.py` isolation note): **117 + 8 =
125 passed.**

## Step 13 — Cache System (verification pass, no new production code needed)

Unlike Steps 8-12, this step did NOT require new production code — every PRD-specified
piece of the cache system (Postgres table since Step 1; Cache Checker/Writer since
Steps 8/10; the flush side effect + `cache_stale_risk` health flag since Step 9; TTL
expiry self-heal since Step 2) had already been built organically while implementing
the query and ingest/delete paths. Confirmed this with a dedicated audit before
proceeding (all 5 PRD "Cache System" bullet points — key normalization, TTL,
invalidation-on-mutation, flush retry/health-flag, self-heal — individually traced
to existing, already-tested code) rather than silently skipping the step.

**What this step actually added: `tests/test_cache_system.py`**, closing verification
gaps that existed only because the pieces were built for OTHER reasons and never
explicitly exercised end-to-end together:
- **Cache invalidation on ingest/delete, proven live, not just "the flush function got
  called":** cache a real query via HTTP, confirm the second identical question hits
  cache, ingest (or delete) an unrelated document, confirm the SAME question now
  misses again — proves the table-wide flush (not scoped to the mutated document)
  actually reaches a real cached row through the full HTTP→graph→Postgres path.
- **TTL self-heal through the REAL Cache Checker node**, not just
  `PostgresAdapter.get_cache` directly (already unit-tested since Step 2): seeded a
  cache row with `expires_at` in the past, sent that exact question through
  `POST /query`, confirmed `cached=False` and the stale stored answer was never
  served — proves `normalize_cache_key` -> lookup -> miss -> re-run-pipeline holds
  end-to-end, not just at the adapter layer.
- **The `cache_stale_risk` flag's full lifecycle over live HTTP**: mocked
  `PostgresAdapter.flush_cache` to raise, ingested a document (ingest itself still
  succeeds — D-16/D-24: a flush failure must never fail the write), confirmed
  `GET /health` reports `cache_stale_risk: true` and overall `status: "degraded"`;
  restored real flush behavior, ingested again, confirmed the flag clears back to
  `false` and status returns to `"healthy"` — the one part of the cache system that
  had never been exercised end-to-end before this step (Step 9 unit-tested the
  Postgres flag primitive in isolation; Step 12 wired `/health` to read it; this step
  is what actually proves the two connect correctly under a real failure/recovery).

**Known, accepted gaps (found during the pre-step audit, deliberately NOT fixed —
documented rather than silently ignored):**
- `_set_flag`'s own write can itself silently fail (`except Exception: pass`,
  `_common.py`) — if BOTH the flush AND the subsequent flag-write fail, `/health`
  would under-report staleness. Accepted for v1: the flag write is a single UPDATE on
  a tiny table, failing independently of a just-failed flush is a low-probability
  double-fault, and the alternative (making a best-effort health signal itself
  retryable/blocking) adds real complexity for a scenario this system's ICL/
  single-tenant scope doesn't warrant guarding harder against.
- Expired cache rows aren't proactively garbage-collected — only removed by the next
  full flush (any KB mutation) or overwritten by a fresh write to the same key. Fine
  at this scale: 48h default TTL, and an actively-maintained SOP wiki mutates often
  enough that rows don't linger meaningfully.
- `normalize_cache_key` has no schema/version prefix — a future normalization change
  would leave old rows silently unmatched (not silently WRONG, just orphaned until
  their own TTL expires) rather than needing an explicit migration. Acceptable; v2
  multi-tenancy (D-29) would need to revisit this anyway (tenant-scoped keys).

**Tests:** 4 new (`test_cache_system.py`, gated on `RUN_GRAPH_INTEGRATION_TESTS=1`
like the rest of the integration suite) — ingest-flush, delete-flush, TTL self-heal
through the real node, and the flag set/clear lifecycle over live HTTP. Full suite
(two processes, per the established isolation note): **121 + 8 = 129 passed.**

## Step 14 — Dead-Letter System (key decisions)

Built directly in this session (Sonnet). Closes the one real gap Step 8's Audit Writer
docstring flagged from the start: exhausting all 3 Postgres retries used to propagate
as a raw exception out of the node, which — since Audit Writer is the LAST node on
every path — would have 500'd an otherwise-successful, already-completed user request
just because its own audit LOG entry failed to write.

**New/changed files:**
```
app/dead_letter.py               # write_dead_letter, replay_dead_letters, has_backlog/poisoned
app/graph/nodes/audit.py          # catches retry-exhaustion, dead-letters instead of raising
app/api/app.py                    # lifespan calls replay_dead_letters() at startup
app/api/routers/health.py         # audit_backlog/audit_poisoned: STUBBED → REAL
app/config.py                     # + dead_letter_path (DEAD_LETTER_PATH, already seeded)
tests/test_dead_letter.py
```

**Design**: two JSON Lines files on `DEAD_LETTER_PATH` (a host folder meant to survive
container restarts — Step 17 wires the actual Docker volume): `dead_letter.jsonl`
(pending entries, each carrying a `retry_count`) and `poisoned.jsonl` (entries that
failed `MAX_REPLAY_ATTEMPTS=3` replay attempts — read for the health flag, otherwise
untouched forever; admin reviews manually per the PRD). `replay_dead_letters()` is the
literal "startup replay routine" the PRD specifies — called once from FastAPI's
lifespan, not on any recurring schedule.

**Audit Writer's failure path, rewritten**: `write_audit`'s 3 retries are unchanged;
what's new is what happens after they're exhausted. Instead of letting the exception
escape the node, it's caught and handed to `write_dead_letter()` — itself wrapped in
its own try/except, since even the dead-letter write can fail (an unwritable volume),
and at that point there's genuinely nothing further this layer can do. The node always
returns `{}` normally either way. The event's timestamp is now computed ONCE up front
(not inside the adapter on every retry attempt) so a replayed entry preserves the
ORIGINAL event time rather than recording when it happened to get replayed.

**Idempotency carries through replay for free**: `write_audit`'s existing `ON
CONFLICT (idempotency_key) DO NOTHING` (Step 1/2) means a replay of an event that
actually DID reach Postgres on some earlier attempt (dead-lettered only because the
CONFIRMATION of that write itself failed, a rare but real race) safely no-ops instead
of duplicating — no new logic needed for this, just relying on what Step 2 already
guaranteed.

**Real verification, not just unit-level mocking** (throwaway Postgres 16 + Qdrant,
real filesystem): forced `PostgresAdapter.write_audit` to fail, confirmed
`audit_writer` returns normally (never raises) and a real file appears on disk with
the correct idempotency_key and `retry_count=0`; confirmed no audit_log row exists yet
in Postgres. Restored real behavior, ran `replay_dead_letters()`, confirmed the entry
disappears from the pending file AND a real audit_log row now exists. Separately,
forced 3 consecutive failed replays and confirmed the entry moves to the poisoned
file with `retry_count=3`, is removed from the pending file, and a 4th replay leaves
it completely untouched (never retried again, per spec). Confirmed `GET /health`
reflects both `audit_backlog` and `audit_poisoned` correctly over real HTTP. Most
importantly: confirmed the FastAPI **lifespan itself** — not just a manual call —
triggers replay automatically (seeded a pending entry, started a brand-new
`TestClient` from scratch, confirmed the backlog was gone by the time `/health`
responded), proving the startup wiring actually works, not just the underlying
function in isolation.

**A same-class testing pitfall as Steps 12/13, found again and fixed the same way**:
running `test_dead_letter.py` alone passed cleanly, but running it as part of the full
suite intermittently failed one test — the same cross-event-loop pooled-connection
issue (pytest-asyncio gives every test function its own event loop by default; the
shared `app.db.session.engine`'s pool is a process-wide singleton). Unlike Steps 12/13
(which only needed to dispose the pool once, right before opening a `TestClient`),
every test in this file touches the shared engine directly — so the fix here is an
`autouse` fixture that disposes the pool before EVERY test in the file, not just the
one that happens to use `TestClient`.

**Tests:** 4 new (`test_dead_letter.py`, gated on `RUN_GRAPH_INTEGRATION_TESTS=1`) —
capture-without-raising, successful replay, poisoning-after-3-attempts (and
never-again), and the health-flag lifecycle over live HTTP. Full suite (two
processes, per the established isolation note): **125 + 8 = 133 passed.**

## Step 15 — Health Endpoint (final flag) (key decisions)

Built directly in this session (Sonnet). 6 of the 7 health flags were already real by
the end of Step 14 (postgres/qdrant connectivity since Step 12, cache_stale_risk since
Step 9/12, audit_backlog/poisoned since Step 14, status aggregation since Step 12).
This step closes the last one: `embedding_model_available`.

**Changed files:**
```
app/embedding.py               # + Embedder.is_loaded public property
app/api/routers/health.py       # embedding_model_available: STUBBED → REAL
tests/test_health_endpoint.py
```

**The real design problem**: the PRD's own Success Criteria requires the health
endpoint to respond "under 2 seconds (including the embedding model check)," but
Nomic is a ~500MB-1GB CPU model (Crash Risk #1) — a cold load can take far longer than
that, and a health check triggering a multi-GB download as a side effect would be a
bad design regardless of timing. Resolution, using Step 6's existing lazy-singleton
`Embedder`:
- **Model never loaded yet in this process** (`Embedder.is_loaded` — new public
  property exposing the existing private `_loaded` flag): report `True` WITHOUT
  attempting a load. This is "no failure observed," not "actively confirmed" — a
  judgment call, but the alternative (loading on the health path) both blows the 2s
  budget on a fresh deploy's very first health check and makes health-checking itself
  responsible for triggering the download.
- **Model already loaded** (a real query has happened, or something else warmed it):
  run one genuinely real `embed_query("ping")`, bounded by `asyncio.wait_for(...,
  timeout=1.5)`. A model that's warm and can't embed two characters within 1.5s is a
  real, meaningful signal (stuck process, resource exhaustion) — exactly the failure
  mode this flag exists to catch, and the one case where a real check is both
  meaningful and fast enough to afford.

**Real verification, not just reasoning about the timeout:** exercised all four
scenarios live in the actual FastAPI app via `TestClient`, checking response time on
every one —
1. Cold (never loaded): `embedding_model_available: true`, ~0.12s, confirmed
   `is_loaded` stayed `False` afterward (no load was triggered as a side effect).
2. Warm + healthy (fake model injected into the real singleton): `true`, ~0.06s.
3. Warm + broken (fake model raises immediately): `false`, overall `status:
   "degraded"`, ~0.07s.
4. **Warm + genuinely hung** (fake model does a real blocking `time.sleep(5)`,
   exactly the failure mode a naive "just await it" implementation would be
   vulnerable to): `false`, and — this is the check that actually matters — elapsed
   time was **~1.6s, not 5s**, proving `asyncio.wait_for`'s timeout genuinely cuts off
   a hung call rather than being decorative. Without this test, a regression that
   silently dropped the timeout wrapper would only surface in production the first
   time the real model actually got stuck.

**Test isolation note:** `get_embedder()` is a process-wide `@lru_cache` singleton
shared across the whole pytest run; these tests mutate its internal `_model`/`_loaded`
state directly to simulate each scenario, which would leak across tests without care.
Added an `autouse` fixture that snapshots and restores the singleton's state around
every test in this file — the same category of fix as Steps 12-14's cross-event-loop
engine-pool issues, just a different kind of shared process-wide state this time.

**Tests:** 4 new (`test_health_endpoint.py`, gated on `RUN_GRAPH_INTEGRATION_TESTS=1`)
covering all four scenarios above, each asserting both the correct flag value AND that
the response stayed under the 2-second budget. Full suite (two processes, per the
established isolation note): **129 + 8 = 137 passed.**

## Step 16 — MCP Server (key decisions)

Built directly in this session (Sonnet). A genuinely new surface for this build — the
first non-FastAPI, non-graph service — and the first step that required real-time
empirical probing of a third-party SDK's actual API before writing any production
code, since the installed `mcp` package (2.0.0) is a much newer major version than
what's commonly documented, with a materially different API shape (`FastMCP` renamed
to `MCPServer`, moved to `mcp.server.mcpserver`, etc.).

**New files:**
```
app/mcp_server/
├── server.py       # MCPServer: llm_wiki_query tool, source_listing resource, auth
│                     # middleware, rate limiting
├── client.py        # thin httpx wrapper — the ONLY network calls in this package
└── errors.py         # MCP {code,message,retryable} shape + FastAPI-error mapping
tests/test_mcp_server.py
```
Added `mcp>=2.0` dep; promoted `httpx` from `[dev]` to a main dependency (client.py
uses it in production, not just tests). Added `mcp_fastapi_base_url`,
`mcp_rate_limit_per_minute`, `mcp_query_timeout_seconds`, `mcp_host`, `mcp_port` to
`Settings`; new `.env.example` block.

**Three real SDK constraints, discovered by writing and running throwaway probe
scripts against the installed SDK BEFORE writing any production code (not by reading
docs, which don't match this major version) — all documented in server.py's own
docstring, not just here:**
1. `mcp.server.fastmcp.FastMCP` doesn't exist in this version; the class is
   `mcp.server.mcpserver.MCPServer`. `Context` must be imported from
   `mcp.server.mcpserver`, NOT `mcp.server.context` — a similarly-named class exists
   at the wrong path and produces a cryptic Pydantic schema-generation crash if used.
2. **Static `@server.resource(...)` handlers cannot receive an injected `Context`** —
   the SDK raises `ValueError` at registration time ("Context injection for static
   resources is not supported"). This directly threatened the design: `source_listing`
   is explicitly a static, non-templated resource per the PRD (MCP-1). Per-tool
   `Context.headers` access (confirmed working for `llm_wiki_query`) couldn't be the
   only auth mechanism, since the resource needs the same per-request key.
3. Resolved via `ServerMiddleware` — confirmed empirically to intercept every request
   type uniformly (`initialize`, `tools/call`, `resources/read`, everything) and to
   expose the real underlying Starlette `Request` (with real headers) via
   `ctx.request`, for BOTH the HTTP/SSE transport's tool and resource paths. The
   middleware extracts `X-API-Key` into a `ContextVar`, which both the tool
   (redundantly, via its own `Context` too) and the resource (its only option) read
   from — one mechanism, uniformly applied, rather than two different ones bolted
   together.

Also found by testing rather than assuming: a bare `-> dict` return-type annotation on
the tool silently produces text-only output (`structured_content` stays `None`) — the
SDK only derives a structured-output JSON schema for `dict[str, Any]` (or a proper
Pydantic model). Fixed by annotating `dict[str, Any]`; verified the fix by checking
`structured_content` is actually populated over a real MCP session, not just that the
text blob (which was always correct) parses as JSON.

**Auth design (Section 4/6): MCP validates nothing itself.** No Postgres connection
exists in this package at all (verified — see below) — the API key is extracted by
the middleware and forwarded as `X-API-Key` on every FastAPI call; FastAPI's own
`require_tier` dependency (Step 12) is the actual, sole source of truth for whether a
key is valid, active, and permitted. A 401 from FastAPI just becomes another error
MCP translates, same as any other. This is the literal, structural meaning of "thin
wrapper" and "zero direct connections" — not just avoided in spirit, but structurally
impossible, confirmed by importing `app.mcp_server.server` and checking `sys.modules`
for anything SQLAlchemy/asyncpg/psycopg/qdrant-shaped: none. That check is now a
permanent regression test (`test_mcp_server_module_never_imports_db_or_vector_store_clients`).

**Error mapping — an honest coverage gap, not hidden:** the PRD lists 7 MCP error
codes, but this implementation only reaches 3 of them (`invalid_input`,
`retrieval_error`, `listing_error`) via explicit mapping; the rest fall to the
documented catch-all (`internal_error`, `retryable: true`). This is a direct,
already-locked consequence of Steps 10/12's design, not a new gap introduced here:
`insufficient_results`, `generation_error`, and `grounding_failed` are all
deliberately surfaced by FastAPI as ordinary 200 responses (informative answer text,
`degraded` flag) rather than HTTP errors — "LLM never blocks retrieval" (Node 7)
extended consistently to "insufficient retrieval doesn't block a response either." A
calling agent still gets a clear, honest answer either way; it just never sees those 3
codes via the `{code,message,retryable}` shape. Reopening Steps 10/12 to manufacture
these codes for the sole benefit of a fuller-looking error table wasn't worth
relitigating a tested decision. `embedding_error`/a full `retrieval_error` are also
only partially reachable, since Steps 8/10 never wrapped Embedder/Retriever node
failures in Generator's style of dedicated try/except — flagged as a pre-existing
query-path robustness gap, not something Step 16 introduced or should silently paper
over.

**Real end-to-end verification — both servers actually running, not mocked at any
layer between them:** ran real FastAPI (Steps 12-15, dense embedder + LLM mocked per
every prior integration test's pattern) and the real MCP server as two live `uvicorn`
processes (background threads), connected with the real `mcp` client library over
HTTP/SSE, and drove the whole stack through a genuine MCP session:
- `llm_wiki_query` end-to-end: real hybrid retrieval, real ranking, real grounding
  verification, structured output matching the PRD's exact 7-field contract.
- `source_listing`: real document listing reflecting an actual ingest.
- Input validation (empty/too-long question) rejected by MCP itself, confirmed
  FastAPI is never even called for these (checked via FastAPI's own request logs).
- A garbage API key correctly produces FastAPI's 401, correctly mapped to the
  documented `internal_error`/`retryable:true` fallback.
- Rate limiting: exactly 10 successful calls then consistent `"Too many requests."`
  rejections, matching `MCP_RATE_LIMIT_PER_MINUTE` precisely.
- The "zero direct connections" claim, checked by inspecting actual imported modules,
  not just by not writing the import.

Found and fixed 4 real bugs during this verification, all in test-script/test-file
plumbing rather than the server itself (documented so the next debugging session
doesn't rediscover them): a background-thread event-loop signal-handler pitfall
(`uvicorn.Server.run()` vs `.serve()` — `.run()` installs signal handlers, which only
works on the main thread); the by-now-familiar cross-event-loop pooled-connection
issue (same `engine.dispose()` fix as Steps 12-15, needed again here since the FastAPI
thread gets its own fresh loop); a hardcoded fake `document_id` in the test's own mock
LLM that had nothing to do with the real citation being verified; and an assertion
that assumed exclusive ownership of the shared Qdrant collection, which doesn't hold
once many other test files' similarly-worded documents are already in it under a
non-discriminating constant-vector fake embedder — relaxed to assert what the test can
actually guarantee (response shape, a grounded/well-formed citation) rather than which
exact document a same-vector-everywhere hybrid search happens to rank first.

**Tests:** 6 new (`test_mcp_server.py`, gated on `RUN_GRAPH_INTEGRATION_TESTS=1`, real
FastAPI + MCP servers as live processes) covering the query tool, the resource, input
validation, auth-error mapping, rate limiting, and the zero-DB-imports guarantee. Full
suite (two processes, per the established isolation note): **135 + 8 = 143 passed.**

## Step 17 — Docker Compose (key decisions)

Built directly in this session (Sonnet). The first step that packages the whole system
for real deployment rather than adding application logic — and the first step where
the majority of the work was fighting genuine infrastructure problems (a corrupted
Docker storage layer from a host disk-full incident, and an accidental multi-GB CUDA
dependency bloat) rather than writing new code. Both are documented in full below,
since they're the actually load-bearing lessons from this step.

**New files:**
```
Dockerfile           # main app image (FastAPI + LangGraph + Nomic, CPU-only torch)
Dockerfile.mcp         # MCP server image — separate, deliberately minimal deps
docker-compose.yml      # 4 services, 3 volumes, healthchecks, startup ordering
.dockerignore
tests/test_docker_compose.py
```
Changed: `app/api/app.py` (lifespan now also runs `ensure_collection()` and
`setup_checkpointer_schema()` at startup — see below); `.env.example` (Docker-appropriate
hostnames + `POSTGRES_ADMIN_USER/PASSWORD/DB` for the Postgres container's own bootstrap).

**Startup bootstrap completed, closing a real gap in the documented setup flow:**
Section 10's Setup Instructions list `docker compose up` → `alembic upgrade head` →
create admin key → ingest → query, with no separate step for provisioning the Qdrant
collection or the checkpointer's own tables. Both `ensure_collection()` (Step 7) and
`setup_checkpointer_schema()` (Step 8) were explicitly designed from the start as
"safe to call on every startup" — so this step finally wires them into the FastAPI
lifespan (`app/api/app.py`), alongside the dead-letter replay routine that was already
there (Step 14). `docker compose up` + `alembic upgrade head` + creating an admin key
is now genuinely the complete bootstrap, matching Section 10 exactly, with no
undocumented manual steps. (Alembic migrations themselves deliberately stay manual —
not run from the lifespan — since they need the admin role that creates the
restricted role in the first place, and schema changes should be a reviewed, explicit
action, not something a container silently does on every boot.)

**MCP gets a genuinely separate, minimal image, not just separate source files:**
`Dockerfile.mcp` installs its own small, explicit dependency list (`mcp`, `httpx`,
`pydantic`, `pydantic-settings`, `uvicorn`) rather than reusing the main app's full
`pyproject.toml` dependencies (which include sqlalchemy/torch/qdrant-client/etc.).
Result: 307MB vs. 600MB for the main app image — the "thin wrapper, zero direct
connections" claim from Step 16 (already verified structurally via import-checking) is
now ALSO backed by what's physically shipped in the image, not just what Python
imports at runtime.

---

**Real infrastructure problem #1 — a corrupted Docker storage layer, traced to the
host disk being completely full.** An early heavy build attempt failed with
`input/output error` writing to BuildKit's containerd metadata store; a naive retry
made `docker images`/`docker system df` themselves start failing with I/O errors on
missing blobs. Traced to the actual root cause rather than just restarting blindly:
`df -h /` showed 153MB free out of 228GB — an APFS container-wide space exhaustion
(shared free-space pool across all volumes on the disk, not just this one). Docker's
virtual disk writes were failing mid-transaction against a full disk, corrupting its
content-addressed blob store. Fixed properly: user freed host disk space (153MB → 22GB
free), then Docker Desktop was restarted (`docker desktop restart`, escalating to a
manual `SIGTERM`/`SIGKILL` + relaunch when the graceful CLI restart itself timed out
against the already-wedged backend) — after which `docker system df` worked cleanly
again and the build proceeded normally. Restarting Docker Desktop stopped an unrelated
container (`ieh-postgres`, from a different project) as a side effect — flagged to the
user rather than silently absorbed, since it wasn't part of this build.

**Real infrastructure problem #2 — torch's default PyPI wheel pulls in the full NVIDIA
CUDA toolkit as transitive dependencies, ~2-3GB of completely unused packages**
(cuDNN, cuBLAS, cuSPARSE, NCCL, Triton, ...). Caught by actually watching the build log
rather than trusting it would finish: nothing in this app runs on GPU — Nomic is an
explicitly in-process CPU model (Crash Risk #1/#4, D-35) — so that entire toolkit
would have sat in the image, unused, multiplying both build time and image size for
zero benefit. Fixed by installing a CPU-only torch build FIRST, from PyTorch's own CPU
wheel index (`--index-url https://download.pytorch.org/whl/cpu`), before installing
the rest of the app — pip then sees torch already satisfied and never reaches for the
CUDA-bundling default. Confirmed live: `torch-2.13.0+cpu` at 155MB vs. the default
build's 427MB, with zero `nvidia_*` packages appearing in the install log at all
(previously: cudnn 444MB + cusparselt 221MB + nccl 206MB + nvshmem 60MB + cublas 542MB
+ more, several GB before even reaching the rest of the app's dependencies). This
wasn't a hypothetical optimization — it was actively happening in the first build
attempt and would have shipped a needlessly multi-gigabyte image.

---

**Real, live, multi-container end-to-end verification** — not a mocked test, the
actual built images, actually deployed via `docker compose up`, exercised over their
published ports from the host:
- `docker compose config` validates cleanly (6 structural tests: 4 services, 3
  volumes, shared network, healthy-dependency ordering, MCP has no volume mounts,
  correct per-service Dockerfile).
- Both images build successfully; final verified sizes via `docker image inspect`
  (the authoritative figure — `docker images`' summary column showed a
  layer-deduplication display quirk not worth chasing further): app 600MB, mcp 307MB.
- `docker compose up -d`: postgres and qdrant reach `healthy` before `app` even starts
  (proving `depends_on: condition: service_healthy` works); `app` correctly FAILS on
  first start (`role "llm_wiki_app" does not exist`) since migrations haven't run yet
  — exactly the intended behavior, not a bug — then `restart: on-failure`
  automatically recovers it the moment `docker compose run --rm app alembic upgrade
  head` creates that role.
- `GET /health` from the host (`localhost:8000`) reports fully healthy, confirming the
  new lifespan bootstrap (Qdrant collection + checkpointer schema) ran successfully
  inside the real container against the real compose-networked Postgres/Qdrant.
- **A real, non-mocked document ingest** (first attempt hit the 120s ingestion timeout
  — a genuine cold Nomic model download+load exceeding the request budget, not a bug;
  the model kept loading in its background thread past the cancelled request, and a
  retry seconds later succeeded in under a second). This is the first time in the
  entire build that a real ingest ran against the real production embedding model
  rather than a mocked one.
- **A real, non-mocked query** against that real ingest: real hybrid Qdrant search
  (genuine combined score 1.078), real ranking, and — since no Anthropic key is
  configured in this environment — the Generator's degraded fallback fired exactly as
  designed in Step 10 (`degraded: true`, PRD's exact fallback string, `source_chunks`
  populated from real `ranked_chunks`). A realistic "LLM not configured" failure mode,
  handled by already-tested code, not a new path built for this step.
- **The real MCP server, in its own separate container**, connected to over its
  published port (`localhost:8001`) with a real `mcp` client: `llm_wiki_query`
  correctly proxied to the real FastAPI app and returned the identical degraded
  result; `source_listing` correctly listed the real ingested document — confirming
  the two containers talk to each other correctly over the compose network exactly as
  configured (`MCP_FASTAPI_BASE_URL=http://app:8000`).
- **Volume persistence, proven, not assumed:** restarted postgres/qdrant/app
  (`docker compose restart`) and confirmed via `GET /documents` that the ingested
  document's Postgres row survived, and via a re-query that its Qdrant vector data
  survived too (same chunk, same content, a fresh real hybrid score) — genuine proof
  the 3 named volumes (`postgres_data`, `qdrant_data`, `dead_letter_data` — the last
  one implicitly verified by the app starting without permission errors on that mount)
  actually persist data across container recreation, not just across process restarts
  within a still-running container.

**Tests:** 6 new (`test_docker_compose.py`) — fast, no Docker daemon build/run
required, validate the compose file's structure via `docker compose config` (services,
volumes, network membership, healthy-dependency ordering, MCP's stateless/no-volumes
property, per-service Dockerfile assignment). The full build-and-run verification
above was done live against a real Docker daemon rather than captured as an automated
test — a multi-minute image build isn't something every CI run should pay for on every
commit; Step 18 revisits test-suite composition and tiers more broadly. Since
`app/api/app.py`'s lifespan changed (adding the `ensure_collection`/
`setup_checkpointer_schema` calls) — code every `TestClient`-based test in the suite
exercises on startup — re-ran the FULL existing suite against fresh throwaway
containers rather than assuming the change was safe: **149 passed** (141 in the main
run + 8 for `test_postgres_adapter.py`, run separately per the established
schema-wiping-fixture isolation note).

## Step 18 — CI Test Suite (key decisions)

PRD Section 9 locks this step's shape precisely: "~52 tests, in-memory Postgres,
mocked Qdrant. 3 per endpoint + specific failure tests. Failing test blocks deploy."
That's a genuinely different suite from everything built so far — Steps 8-17's tests
(the "149 passed" figures throughout this log) are real, opt-in integration tests
against live Postgres+Qdrant Docker containers, gated behind
`RUN_GRAPH_INTEGRATION_TESTS=1`, built alongside each feature specifically to catch
real cross-system bugs (and did — the Step 9 SQLAlchemy staleness bug and both Step 11
routing bugs would not have been caught by mocks). This step is the fast, always-on
deploy gate instead: no live Qdrant, no live checkpointer container, no model
downloads, ~10s wall clock.

**"In-memory Postgres"** — there's no real in-memory Postgres product (unlike
SQLite's `:memory:` mode), and this schema is genuinely Postgres-specific (native
UUID columns, JSONB, `INSERT ... ON CONFLICT DO NOTHING/UPDATE`) in ways SQLite
can't run unmodified — swapping the adapter layer's dialect just for this test suite
would be a bigger, riskier change than the step calls for. Resolved as a real,
throwaway `postgres:16` container (`scripts/run_ci_tests.sh`) with its data directory
on `--tmpfs` (RAM-backed, never touches disk) and
`fsync=off -c full_page_writes=off -c synchronous_commit=off` — genuinely as fast as
an in-memory database in practice, zero schema-compatibility risk, same real grants
and roles Alembic's migration 0001 creates in production. Same "export DATABASE_URL
before Python ever imports app code" discipline used for every Docker-backed test
since Step 8, since `app.db.session.engine` is a module-level singleton built from
`DATABASE_URL` at import time — the script starts the container, waits for
`pg_isready`, exports the URLs, runs `alembic upgrade head`, *then* invokes pytest.

**Mocked Qdrant** — a `FakeQdrantClient` (plain in-memory dict, keyed by point id,
built from real `qdrant_client.models` types) is wired in underneath the real,
unmodified `VectorStoreAdapter` by monkeypatching `build_vector_store` in the three
node modules that call it (ingest/query/delete path) — so the adapter's own
payload-shape and upsert-status assumptions are still exercised for real, only the
network call to an actual Qdrant server is faked. The dense embedder and BM25 sparse
encoder are faked the same way as every integration test since Step 9 (no
~500MB-1GB Nomic download, no fastembed download). `ensure_collection` (Step 7's
startup provisioning, wired into the lifespan since Step 17) is monkeypatched to a
no-op — a real call would try to reach the same Qdrant this suite deliberately never
runs, and would otherwise fail every single test at `TestClient.__enter__`.

**Two things fixed while writing this file — both wrong assumptions in the new test
code, not production bugs, both caught by actually running the suite rather than
reasoning about it:**
1. `DELETE /changelog/{id}` is genuinely NOT idempotent — a plain Postgres adapter
   call, so a second delete of an already-gone row is a real 404. Initially written
   as idempotent by analogy with `DELETE /documents/{id}`, which really is
   deliberately idempotent (it routes through the graph's Deleter node, a Step 11
   design choice documented in documents.py). Fixed by reading changelog.py's actual
   handler rather than assuming symmetry.
2. `RateLimitMiddleware`'s `_buckets` dict lives on a middleware instance that
   Starlette builds once and caches on `app.middleware_stack`, and `app` is a
   module-level singleton shared by every test in the process (documented since Step
   12's test_api.py). This suite has a dozen separate "no API key" tests spread
   across many endpoint classes, all sharing ONE "anonymous" bucket — the later ones
   started seeing 429 instead of 401 once the shared 60s window filled from earlier
   tests. Fixed at the root rather than loosening assertions to tolerate either code:
   the `api_client` fixture now sets `app.middleware_stack = None` before each test,
   forcing Starlette to rebuild the whole stack (and hand out a fresh, empty-bucketed
   `RateLimitMiddleware`) on the next request — genuine per-test isolation, every 401
   assertion stays strict and honest.

**Tests:** `tests/test_endpoints.py` — 52 tests (3 per endpoint × 14 endpoints = 42,
plus 11 specific-failure tests: duplicate-document 409, unsupported-extension 415,
oversized-file 413, insufficient-results degraded-free answer, LLM-failure degraded
answer, cache hit, identical-401-bodies-across-different-rejection-reasons, revoked
key, rate-limit-then-recover, disjoint cursor pagination). Run via
`./scripts/run_ci_tests.sh`: 52/52 passed in ~10s against a throwaway tmpfs Postgres,
run twice back-to-back to confirm reproducibility and clean container teardown
(`trap cleanup EXIT`; confirmed no leftover containers via `docker ps -a` afterward).

## File Tree (additions in Step 7)

```
app/qdrant_setup.py        # ensure_collection() provisioning (dense text 768/Cosine + sparse bm25/IDF)
app/embedding.py           # + BM25Encoder, get_bm25_encoder (shared sparse encoder)
tests/test_qdrant_setup.py
```

## File Tree (additions in Step 6)

```
app/embedding.py           # Embedder (lazy Nomic load), embed_query / embed_chunks, retry + 768 validation
tests/test_embedding.py
```

## File Tree (additions in Step 5)

```
app/chunker.py             # chunk_document() + tiktoken sizing, 3-tier split, overlap, enrichment
app/domain.py              # + Chunk
tests/test_chunker.py
```

## File Tree (additions in Step 4)

```
app/parsers/
├── __init__.py            # registry + parse_document() dispatch + exports
├── base.py                # Block, BlockType, ParsedDocument, FileParseError, PARSE_ERROR_MESSAGE
├── pdf.py                 # parse_pdf — PyMuPDF, blocks + page numbers
├── docx_parser.py         # parse_docx — python-docx, heading styles → levels
├── txt.py                 # parse_txt — blank-line paragraph detection
└── markdown.py            # parse_md — ATX headings + fence-aware paragraphs
tests/test_parsers.py
```

## File Tree (additions in Step 3)

```
app/adapters/llm/
├── __init__.py            # build_llm_adapter() factory + exports
├── base.py                # LLMAdapter ABC, ANSWER_SCHEMA, errors, parse_generation
├── claude.py              # ClaudeAdapter — anthropic, forced tool_use
├── openai_provider.py     # OpenAIAdapter — openai, response_format json_schema
└── llama.py               # LlamaAdapter — openai SDK + base_url, prompt enforcement
app/domain.py              # + GenerationResult
tests/test_llm_adapter.py
```

## File Tree (additions in Step 2)

```
app/
├── domain.py                  # RetrievedChunk, Citation, SparseVector, ChunkWithVectors, ScoredChunk
└── adapters/
    ├── __init__.py
    ├── errors.py              # AdapterValidationError
    ├── vector_store.py        # VectorStoreAdapter (only module importing qdrant-client)
    └── postgres.py            # PostgresAdapter (only runtime module importing ORM/session)
tests/
├── __init__.py
├── test_vector_store_adapter.py
└── test_postgres_adapter.py
```

---

## Build Step Checklist

- [x] **Step 1** — Postgres schema (5 tables), async SQLAlchemy 2.0, Alembic from day
      one, two DB roles with least-privilege grants. DONE & verified against real PG.
- [x] **Step 2** — Adapter-wall: VectorStore (Qdrant) + Postgres adapters, output
      validation, never-retry. DONE & tested (8 mocked + 6 real-PG, all green).
- [x] **Step 3** — LLM adapter: Claude (tool_use) / OpenAI (response_format) / Llama
      (prompt enforcement), standardized `GenerationResult`, config-only swap, never-retry,
      typed errors. DONE & tested (8 mocked, all green).
- [x] **Step 4** — File parsers: PDF (PyMuPDF), DOCX, TXT, MD behind an extension registry,
      shared `ParsedDocument`/`Block` IR, typed `FileParseError`. DONE & tested (11, all green).
- [x] **Step 5** — Chunker: structure-aware 3-tier split (heading/paragraph/sentence),
      tiktoken sizing (500/50), title+heading enrichment with separate citation text,
      deterministic chunk_id. DONE & tested (8, all green).
- [x] **Step 6** — Embedding: in-process Nomic (sentence-transformers, lazy singleton),
      `embed_query`/`embed_chunks`, ~50 batching, 1 retry, 768-dim validation, injectable
      model boundary. DONE & tested (9, mocked model; real load not downloaded in sandbox).
- [x] **Step 7** — Qdrant setup: idempotent `ensure_collection` (dense text 768/Cosine +
      sparse bm25/IDF), shared `BM25Encoder` (fastembed Qdrant/bm25). DONE & verified against
      real Qdrant + real BM25 (7 tests, all green).
- [x] **Step 8** — LangGraph state schema + graph wiring: 19-field state, sealed-envelope
      RunnableConfig, all 14 nodes (8 real, 6 stubbed for Steps 9-11), 4 PRD routing
      points + 2 gap-fill branches, AsyncPostgresSaver checkpointer (verified real,
      least-privilege grants resolved). DONE & tested (25 tests incl. real-DB
      integration; full suite 82 passed).
- [x] **Step 9** — Ingestion path: Storer filled in for real (delete-first clean slate
      #21, Postgres-then-Qdrant with rollback #24, 1 retry, cache-flush side effect +
      new `health_flags` table). Found/fixed 2 real bugs (Step 8's missing import;
      SQLAlchemy identity-map staleness on upsert, also retrofixed in Step 2's
      write_cache). DONE & tested (real ingest + duplicate + rollback verified against
      live Postgres+Qdrant; 86 passed).
- [x] **Step 10** — Query path: Ranker (weighted sum + recency tiebreak via new
      Qdrant-payload ingested_at), Quality Gate (threshold), Generator (LLM adapter +
      3-layer grounding verification + degraded fallback), Cache Writer, all filled in
      for real. DONE & tested (real ingest→query→cache-hit verified against live
      Postgres+Qdrant with a mocked LLM; 96 passed).
- [x] **Step 11** — Delete/Update path: Deleter filled in for real; update path's
      old/new document_id split resolved (Chunker stashes previous_document_id,
      Deleter targets it). Found/fixed 2 real bugs via live end-to-end testing: Storer
      KeyError on every update (content_hash never computed for that path), and a
      failed update silently deleting the old document anyway (routing didn't check
      Storer's status) — exactly the "hole" D-2 exists to prevent. DONE & tested (live
      update+FK-SET-NULL+delete+idempotency+negative-test verified; 102 passed).
- [x] **Step 12** — FastAPI endpoints: all 14 routes, auth (dependency-based, key
      table lookup), rate limiting per tier (in-memory, before-auth ordering verified
      live), tiered timeouts, error handlers, cursor pagination, admin key CLI. Found
      2 real bugs (FastAPI silently drops 204 bodies — fixed with manual
      JSONResponse; PUT /documents/{id} shared the 120s ingest timeout instead of the
      30s update timeout). DONE & tested (full HTTP surface verified live; 125 passed).
- [x] **Step 13** — Cache system: already fully built organically (Steps 1/2/8/9/10);
      this step audited it against all 5 PRD bullet points and closed the remaining
      end-to-end verification gaps (invalidation-on-mutation, TTL self-heal through
      the real node, cache_stale_risk flag set/clear lifecycle over live HTTP). No new
      production code. DONE & tested (129 passed).
- [x] **Step 14** — Dead-letter system: JSONL pending/poisoned files on
      DEAD_LETTER_PATH, Audit Writer now dead-letters instead of raising past its 3rd
      retry (fixing a gap flagged since Step 8 — a failed audit log write no longer
      500s an already-successful user request), startup replay routine wired into
      FastAPI's lifespan, GET /health's audit_backlog/audit_poisoned now real. DONE &
      tested (full capture->replay->poison lifecycle verified live, incl. the lifespan
      trigger itself; 133 passed).
- [x] **Step 15** — Health endpoint: all 7 flags now real. Last one
      (embedding_model_available) designed around the PRD's own "<2s including the
      embedding check" constraint — optimistic when the model was never loaded (no
      cold-load side effect on the health path), a real bounded-timeout ping when it
      was. DONE & tested (all 4 scenarios incl. timeout-cutoff verified live with
      response-time assertions; 137 passed).
- [x] **Step 16** — MCP server: llm_wiki_query tool + source_listing resource, both
      thin HTTP wrappers around FastAPI (zero DB/vector-store imports, verified by
      module inspection). Auth via ServerMiddleware (static resources can't use
      Context injection in this SDK version — discovered empirically), per-key rate
      limiting, FastAPI-error-to-MCP-code mapping (3 of 7 PRD codes reachable; rest
      fall to the documented internal_error fallback, a direct consequence of Steps
      10/12's graceful-degradation design, not a new gap). DONE & tested (both real
      servers run live as separate processes and driven by a real MCP client;
      143 passed).
- [x] **Step 17** — Docker Compose: 4 services (app/postgres/qdrant/mcp), 3 volumes,
      healthcheck-gated startup ordering, separate minimal MCP image (307MB vs app's
      600MB). Closed a real setup-flow gap (Qdrant collection + checkpointer schema
      now bootstrap automatically in the app's lifespan, matching Section 10 exactly).
      Fixed 2 real infra problems found by actually building/running: a host-disk-full
      incident that corrupted Docker's storage (traced, fixed, documented — not a code
      issue), and torch's default wheel silently pulling ~2-3GB of unused NVIDIA CUDA
      packages (fixed with a CPU-only torch install, confirmed 155MB vs 427MB with
      zero nvidia_* packages). DONE & tested (real multi-container docker-compose up,
      real non-mocked ingest+query+MCP round trip, real volume-persistence-across-
      restart proof; 149 passed after re-running the full suite for the lifespan change).
- [x] **Step 18** — CI test suite: `tests/test_endpoints.py` (52 tests — 3 per
      endpoint × 14 endpoints + 11 specific-failure tests) plus
      `scripts/run_ci_tests.sh`, the deploy-blocking gate script. Deliberately
      distinct from Steps 8-17's opt-in `RUN_GRAPH_INTEGRATION_TESTS=1` suite (real
      containers, built to catch cross-system bugs as each feature landed) — this one
      is the fast, always-on CI gate the PRD actually specifies: "~52 tests,
      in-memory Postgres, mocked Qdrant."
        - **"In-memory Postgres" judgment call**: there's no real in-memory Postgres
          product (unlike SQLite's `:memory:`), and the schema leans on
          Postgres-specific types (native UUID, JSONB, `ON CONFLICT`) SQLite can't run
          unmodified — swapping dialects just for tests would be a bigger, riskier
          change than this step asks for. Resolved as a real, throwaway `postgres:16`
          container with its data directory on `--tmpfs` (RAM-backed) and
          fsync/full_page_writes/synchronous_commit off: genuinely as fast as
          in-memory in practice, zero schema-compatibility risk. Same "export
          DATABASE_URL before Python ever imports app code" pattern used for every
          Docker-backed test since Step 8 (`app.db.session.engine` is a module-level
          singleton built at import time).
        - **Mocked Qdrant**: a `FakeQdrantClient` (in-memory dict, real
          `qdrant_client.models` types) swapped in under the real `VectorStoreAdapter`
          by monkeypatching `build_vector_store` in the 3 node modules that call it
          (ingest/query/delete path) — so the adapter's own payload/status
          assumptions are still exercised for real, only the network call is faked.
          The dense embedder and BM25 sparse encoder are faked the same way as every
          prior integration test (no ~500MB-1GB Nomic download, no fastembed
          download). `ensure_collection` (Step 7's startup provisioning) is
          monkeypatched to a no-op, since a real call would try to reach the same
          Qdrant this suite deliberately never runs.
        - **Real bugs/false assumptions found writing this suite** (none were
          production bugs — all were the test file's own wrong assumptions, caught by
          actually running it against real Postgres rather than guessing):
          1. `DELETE /changelog/{id}` is genuinely NOT idempotent (plain Postgres
             adapter call — a second delete of an already-gone row is a real 404),
             unlike `DELETE /documents/{id}` (deliberately idempotent because it
             routes through the graph's Deleter node). Initially wrote both as
             idempotent by analogy; fixed by reading changelog.py's actual handler.
          2. `RateLimitMiddleware`'s `_buckets` dict lives on a middleware instance
             Starlette builds once and caches on `app.middleware_stack` — and `app` is
             a module-level singleton shared by literally every test in the process
             (documented since Step 12's test_api.py). A suite with a dozen separate
             "no API key" tests spread across many test classes shares ONE
             "anonymous" bucket, so the later ones started seeing 429 instead of 401
             once the shared window filled. Fixed at the root rather than papering
             over it with tolerant assertions: the `api_client` fixture now sets
             `app.middleware_stack = None` before each test, forcing Starlette to
             rebuild the stack (and therefore hand out a fresh, empty-bucketed
             `RateLimitMiddleware`) on the next request — genuine per-test isolation,
             every 401 assertion stays strict.
      DONE & tested: `./scripts/run_ci_tests.sh` — 52/52 passed in ~10s against a
      throwaway tmpfs Postgres, run twice back-to-back to confirm reproducibility and
      clean container teardown (`trap cleanup EXIT`; confirmed no leftover containers
      via `docker ps -a` after each run).
- [x] **Step 20** — Code-debugger validation pass. Extensive live manual testing (not
      just automated tests) surfaced and fixed 9 real bugs, each with a regression
      test and live verification against the running Docker stack:
        1. Changelog FK violation (`document_id` pointing at a nonexistent document)
           returned a raw 500 instead of 404.
        2. `DELETE /documents/{id}` and `DELETE /changelog/{id}` claimed a 204 body in
           code but uvicorn strips 204 bodies per HTTP spec — dead code, fixed to a
           plain empty `Response(204)`.
        3. Idempotent-204-vs-404 on a double `DELETE /documents/{id}` — reviewed,
           decided to KEEP the existing idempotent-204 behavior (no code change).
        4. Malformed UUID (path params, body fields, cursor) and a stale/nonexistent
           cursor both 500'd instead of 400 — added `require_valid_uuid()` /
           `AdapterValidationError` handling across documents/changelog/audit routers.
        5. Whitespace-only `title`/`source_label` accepted on ingest; empty `entry`
           accepted on `PUT /changelog/{id}` (asymmetric with `POST`'s validation) —
           both fixed.
        6. No key-revocation mechanism existed at all despite the PRD's `active` flag
           and Success Criteria promising it — built `app/revoke_api_key.py` +
           `PostgresAdapter.set_api_key_active()`; `create_api_key.py` now also prints
           `key_id`, the only handle available to revoke a key later.
        7. `VectorStoreAdapter.search()`'s dense+sparse merge could exceed `top_k`
           (confirmed live: 7 chunks returned against a configured `top_k=5`, and
           confirmed via audit log that this had silently affected 4 of the earlier
           "clean" grounding tests too) — fixed by truncating in the Ranker,
           immediately after sorting.
        8. Generator's system prompt had no instruction to disclose an embedded
           prompt-injection attempt in a retrieved chunk — it happened to disclose one
           anyway on its own judgment once, but nothing required it. Added an explicit
           instruction; added both a fast deterministic CI test (prompt contains the
           instruction) and an opt-in real-Anthropic-API test
           (`RUN_GRAPH_INTEGRATION_TESTS=1`, the only test in the suite that costs
           real money, documented as such).
        9. An exhausted `EmbeddingError` (query embed or ingest batch embed) had no
           handler anywhere between the node and FastAPI — raw 500. Also, once this
           model entered a corrupted state (a real, reproducible rotary-embedding
           tensor-size-mismatch bug — see Known Limitations below), `/health` kept
           reporting `embedding_model_available: true` because its ping probe used a
           short, unrepresentative string that could coincidentally avoid the
           corrupted length. Fixed: `embedder_query`/`embedding_batcher` now catch
           `EmbeddingError` and route to Audit Writer (new conditional edges,
           `route_after_embedder_query`/`route_after_embedding_batcher`, mirroring the
           existing `route_after_duplicate_checker` gap-fill pattern) instead of
           crashing the next node on a missing state key; `error_mapping.py` maps it
           to a proper `503`; the health check now tracks real embed-call failures in
           a timestamped sliding window (`Embedder.recently_failed`, same pattern as
           `RateLimitMiddleware` — race-safe under concurrent calls, since a
           concurrent success must never silently erase evidence of a real failure)
           and checks that before falling back to the ping.
      Beyond the 9 fixes, extensive additional manual testing passed clean: 6 prompt
      injection variants (direct exfiltration, DAN-style jailbreak, fake-authority
      override, false-fact-confirmation, indirect/document-embedded injection via an
      uploaded poisoned doc, conflicting-documents disclosure) — all refused or
      correctly grounded, zero hallucination, zero compliance. MCP path independently
      verified live (real MCP client, real SSE server) to expose identical data to the
      raw HTTP path — reviewed and deliberately left as-is (citations already expose
      the same chunk content on every successful query; MCP-3 locks single-tier
      full-KB read access by design). Dead-letter/failure-recovery cycle verified
      fully live (real Postgres permission revoked → request still succeeds →
      dead-letter file captures the exact entry → `/health` flags `audit_backlog` →
      permission restored → app restart replays → flag clears → entry confirmed in
      Postgres with its ORIGINAL timestamp preserved). `/health` while Qdrant is
      stopped correctly reports `qdrant_connected: false` / `status: degraded`, not a
      false healthy.

## Known limitations surfaced during Step 20 testing (not fixed — documented)

- **Nomic embedding model can enter a corrupted, persistent state under real use.**
  A real, repeatedly-reproduced `RuntimeError: The size of tensor a (N) must match
  the size of tensor b (M)` inside the model's rotary-embedding code, triggered by
  real traffic (not a specific input — same content sometimes fails, sometimes
  doesn't). Once triggered, EVERY subsequent embed call in that process fails
  identically until the container is restarted — not transient. Failure mode itself
  is now handled safely (finding #9 above: clean 503, health check correctly reports
  it, no raw crash) — but the root cause inside the model/sentence-transformers stack
  is NOT fixed and was NOT isolated (occurrence rate got noticeably worse — close to
  every-other-call — over the course of a very long testing session; possibly
  cumulative host-level resource degradation rather than a purely deterministic
  per-input trigger).
- **Missing `--init` in the Dockerfile/compose config.** Discovered directly: a
  `docker compose restart app` failed outright with `container PID N is zombie and
  can not be killed. Use the --init option...`. Root cause: `uvicorn` runs as PID 1
  with no init process to reap zombie child processes (likely orphaned from
  `asyncio.to_thread` calls tied to the embedding corruption above). Recovered via
  `docker compose up -d --force-recreate app` (safe — stateless container, no data
  loss). Needs `init: true` added to the `app` service in `docker-compose.yml` (or
  `--init` in the Dockerfile) before production. Not fixed this session — flagged as
  a pre-production TODO.
- **Generator can occasionally produce a schema-invalid structured response under
  high-density prompts.** Observed specifically when several retrieved chunks are
  bundled together and one contains an embedded prompt-injection attempt while others
  contain conflicting information from multiple documents — Claude's tool_use
  response has come back missing the required `citations` field entirely. The
  existing 3-layer grounding defense catches this correctly every time observed (degrades
  to the fixed fallback message, no hallucination, no bad citation ever reached a
  caller) — but the real answer text (which appeared to already contain a correct
  injection-disclosure) gets discarded before it can be seen. Root cause not
  isolated — chunk count, prompt length, and the specific injected+conflicting
  content mix are all still open as contributing factors. Deliberately left
  unfixed rather than patching against an unconfirmed cause.

## Session handoff — 2026-08-11, mid Step 19

**Done:** Step 20 fully complete (above). README.md drafted at the repo root
(system description, Mermaid flowchart reflecting the ACTUAL current graph — 14
nodes + 7 conditional routing points, including the 2 new Step 20 gap-fill branches
`route_after_embedder_query`/`route_after_embedding_batcher` — Mermaid sequence
diagram, setup instructions, a Known Limitations section, V2 upgrade paths) — but
**not yet shown to / confirmed by the user**, and does not yet include the `--init`
limitation (needs adding — see above). Two extra verification tests beyond the
original Step 20 scope both PASSED: dead-letter/failure-recovery cycle, and
`/health` correctness while Qdrant is stopped.

**Blocked / in progress:** real multi-page pagination test (upload 22+ documents,
confirm no duplicate/skipped rows walking cursor-based pages). Was at 17 of 22
needed documents when the session paused — blocked by the embedding-corruption bug
above recurring with increasing frequency as the session went on. A
`docker compose restart` even hit the zombie-process issue once, requiring
`--force-recreate` to recover. User was choosing between "keep grinding to 22+" vs
"use the 17 already uploaded with `limit=10` instead of the default 20" when the
session was paused to start fresh.

**Recommended next steps, in order:**
1. Restart the whole Docker stack cleanly (`docker compose down` then
   `docker compose up -d`, or restart Docker Desktop itself first) before resuming —
   the embedding corruption's rate got measurably worse over this session's length,
   suggesting cumulative host-level degradation, not just in-process state a
   container restart alone reliably clears.
2. Finish the pagination test (decide 17-with-smaller-page-size vs push to 22+).
3. Add the `--init` limitation to README.md's Known Limitations section (drafted
   above, not yet copied in).
4. Show the README to the user for review/confirmation (was about to publish as an
   Artifact for Mermaid rendering when the session paused).
5. Commit.
