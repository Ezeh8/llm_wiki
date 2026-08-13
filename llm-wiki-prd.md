# LLM WIKI — PRODUCT REQUIREMENTS DOCUMENT

**Build:** LLM Wiki — Production RAG Knowledge Base for Enterprise SOPs
**Date:** August 6, 2026
**Pipeline:** Router → LangGraph → MCP → FastAPI → Sparring → PRD → Logic-to-Code → Code-Debugger
**Implementation language:** Python only

---

## Section 1: Problem & Purpose

```
Problem:        Enterprise teams have no single, queryable source of truth for SOPs,
                policies, and procedures — knowledge is scattered across documents nobody reads.
User:           Both. Human admin manages the knowledge base. AI agents query it for grounded answers.
Success:        Within 30 days of deployment: 90%+ grounding accuracy, zero ungrounded citations
                pass verification, 20+ documents ingested, agents actively querying.
Failure:        Ungrounded answers served as fact. Data corruption on update/delete.
                Stale cache serving outdated information without detection.
Scope:          ICL (internal tooling). Single tenant per deployment.
Performance:    Query responses under 11 seconds. Cache hits under 1 second.
                Ingestion has no latency target — admin task with 120 second timeout ceiling.
```

---

## Section 2: Agent-Native Layer

```
Agent discoverable:    Yes — via MCP server
MCP server required:   Yes
Agent catalog:         Query tool + Source Listing resource
M2M payments:          No
Discovery endpoint:    None (ICL — agents configured directly)
```

---

## Section 3: LangGraph Architecture

### State Schema (19 fields)

All fields overwrite. None values are routing signals. document_file stored as base64 string. Timestamps as ISO-8601 UTC strings.

| Field | Type | Purpose |
|-------|------|---------|
| operation_type | str | "query" / "ingest" / "delete" / "update" |
| thread_id | str | {operation_type}_{uuid4}, generated at FastAPI boundary |
| session_id | str | Groups related queries. Optional from caller, auto-generated if omitted |
| query_text | str | The question asked |
| document_file | str or None | Base64 file content. Set to None after Chunker finishes |
| document_metadata | dict | {title, source_label, changelog_id (optional)} |
| document_id | str | UUID for the document |
| actor | str | Who performed the action (from API key lookup) |
| chunks | list[dict] | Output of Chunker — raw text chunks with metadata |
| embeddings | list[list[float]] | Output of Embedding Batcher — vectors for chunks |
| retrieved_chunks | list[RetrievedChunk] | Output of Retriever — chunks with scores |
| ranked_chunks | list[RetrievedChunk] | Output of Ranker — re-ranked chunks |
| answer | str | Generated answer text |
| citations | list[Citation] | Verified citations |
| status | str | "success" / "error" / "insufficient" / "cache_hit" / "degraded" |
| error | str or None | Error message if failed |
| cache_hit | bool | Whether the cache was hit |
| cache_key | str | Normalized hash of the query |
| query_embedding | list[float] | Embedding of the query text |

### Pydantic Models

```python
class RetrievedChunk(BaseModel):
    chunk_id: str
    chunk_text: str
    document_id: str
    document_title: str
    score: float
    chunk_index: int

class Citation(BaseModel):
    document_id: str
    document_title: str
    chunk_id: str
    chunk_text: str
    chunk_index: int
```

### RunnableConfig (sealed envelope — NEVER on the state blackboard)

These are passed via RunnableConfig, never stored in state:
- API keys (embedding, vector DB, LLM)
- DB connection strings
- tenant_id (reserved for v2)
- env_mode (dev/demo/production)
- model config (name, temperature, max_tokens)
- file storage path
- vector DB collection name
- chunking config (size=500, overlap=50)
- embedding model name
- cache TTL (default 48 hours)
- hybrid search weights (dense=0.7, sparse=0.3)
- quality threshold (0.65)
- top_k (5)

### RunnableConfig Tags/Metadata (built ONCE at FastAPI boundary)
- thread_id
- session_id
- operation_type
- actor
- env_mode
- app_version

### Node Inventory (14 nodes)

1. **Validator** — Validates input. Copies thread_id to state. Rejects bad file types, empty files, files with no extractable text. Checks file extension (.pdf, .docx, .txt, .md only). Max file size 20MB.

2. **Cache Checker** — Checks Postgres cache by normalized question hash (lowercase, strip punctuation, collapse whitespace, SHA-256). Cache hit → sets cache_hit=True, populates answer + citations from cache, routes to Audit Writer. Cache miss → routes to Embedder. Best-effort (0 retries).

3. **Embedder (query)** — Embeds query text via Nomic-embed-text-v1.5 (768 dimensions). Stores result in query_embedding. 1 retry on failure.

4. **Retriever** — Hybrid search against Qdrant. Dense vector search (Nomic embedding) + sparse vector search (BM25 via Qdrant native sparse vectors). Top-5 results. Optional metadata filtering by source_label and/or document_id from query filter field. Returns list of RetrievedChunk. 1 retry on failure.

5. **Ranker** — Combines dense and sparse scores via weighted sum (0.7 dense / 0.3 sparse, configurable). Applies recency tiebreaker (prefer newer documents on score tie). Pure logic, no external calls.

6. **Quality Gate** — Checks if best chunk score is above 0.65 threshold (configurable). If sufficient → route to Generator. If insufficient → set status="insufficient", route to Audit Writer (skips Generator and Cache Writer). Pure logic.

7. **Generator** — Single LLM node. Claude Sonnet default (eval compares GPT-4o-mini, Llama 3.1 8B). Formatter role ONLY — no training-data knowledge.

   Prompt constraints:
   - Answer ONLY from provided chunks
   - Do NOT add training-data knowledge
   - Cite chunk_id for every claim
   - If chunks don't fully answer, state what's missing
   - Prefer newer source on conflict between chunks
   - Return structured JSON output: {answer: string, citations: [{document_id, document_title, chunk_id, chunk_text, chunk_index}]}

   Uses structured output (tool_use for Claude, response_format for OpenAI, prompt enforcement for Llama). LLM adapter owns translation per provider.

   Post-generation verification: check every citation's chunk_id against ranked_chunks. Strip any citation referencing a chunk_id not in ranked_chunks. If ALL citations stripped → set answer to "I found relevant documents but couldn't generate a verified answer", set citations=[], populate source_chunks with raw ranked_chunks, set degraded=True.

   Citation parsing wrapped in try/catch. Parse failure → same degraded fallback.

   LLM failure → set answer to fallback message, populate source_chunks, set degraded=True. LLM never blocks retrieval.

8. **Cache Writer** — Writes answer + citations to Postgres cache table. Key = normalized hash. 0 retries (best-effort).

9. **Duplicate Checker** — Checks content-hash + title against existing documents in Postgres. If duplicate → set status="error", error="Duplicate document". 1 retry on DB call.

10. **Chunker** — Structure-aware chunking. 500 tokens, 50 token overlap (10%). Split on headings/sections first, paragraph boundaries within oversized sections, sentence boundaries as last resort. Never breaks mid-sentence. Configurable via RunnableConfig.

    Supports 4 file types with dedicated parsers:
    - PDF: PyMuPDF or pdfplumber
    - DOCX: python-docx
    - TXT: text splitting with paragraph detection
    - Markdown: header-aware splitting on # markers

    Each parser wrapped in try/catch. Parse failure → return error "Could not read this file. It may be corrupted or password-protected." (status 422)

    Context enrichment: prepend document_title + section heading (if available) to chunk text before embedding. Store original text for citations.

    State cleanup: sets document_file = None after extracting text and creating chunks. Prevents checkpoint bloat.

11. **Embedding Batcher** — Batches ~50 chunks at a time through Nomic embedding model. Backoff for rate limits. 1 retry per batch. Validates output dimensions (768) before returning.

12. **Storer** — Stores chunks to vector DB and metadata to Postgres.
    - FIRST: delete all existing chunks for this document_id from Qdrant (clean slate per attempt, prevents duplicate chunks on retry)
    - SECOND: write document metadata to Postgres (upsert by document_id)
    - THIRD: write chunks to Qdrant (dense + sparse vectors per chunk, deterministic chunk_id = document_id + chunk_index)
    - If Qdrant write fails after Postgres succeeded: rollback — delete the Postgres metadata row. No ghost documents.
    - 1 retry on failure.
    - Side effect: flush cache (invalidate all cached answers). Cache flush gets 1 retry. If flush fails, set health flag cache_stale_risk=true. Logged by Audit Writer.

13. **Deleter** — Removes document from system.
    - FIRST: delete chunks from Qdrant by document_id
    - SECOND: delete metadata from Postgres
    - Naturally idempotent (deleting something already gone = no-op)
    - Side effect: flush cache (same as Storer — 1 retry, health flag on failure)
    - Changelog FK: SET NULL on document deletion (changelog entries survive with blank link)

14. **Audit Writer** — Append-only insert to Postgres audit table. Fields: event_id, event_type, document_id, query_text, chunks_retrieved, answer_text, actor, timestamp (UTC), status, error_detail. Idempotency key: thread_id + event_type + DB unique constraint. 3 retries. On failure → dead-letter (persistent volume). Dead-letter has retry counter per entry. After 3 failed replays → moved to poisoned file. Never replayed again. Health endpoint flags audit_poisoned=true.

### Routing (4 decision points)

1. **After Validator** → by operation_type:
   - query → Cache Checker
   - ingest → Duplicate Checker
   - delete → Deleter
   - update → Chunker (new document first — ingestion path)

2. **After Cache Checker** → by cache_hit:
   - hit → Audit Writer (skip entire pipeline)
   - miss → Embedder

3. **After Quality Gate** → by status:
   - sufficient → Generator → Cache Writer → Audit Writer
   - insufficient → Audit Writer (skip Generator and Cache Writer)

4. **After Storer (update path)** → Deleter (delete OLD document after new one confirmed stored)
   **After Deleter (delete path)** → Audit Writer
   **UPDATE ORDER: ingest new FIRST, verify success, THEN delete old. Duplicate (temporary) better than a hole.**

### Graph Structure

- Single graph (StateGraph), four operation paths
- All operations share: Validator, Audit Writer, state schema, checkpointer
- Checkpointer: AsyncPostgresSaver (same Postgres), fault-tolerance only
- No fan-out (v2), no streaming (v2), no HITL interrupts

### Adapter-Wall

Two adapters. Graph NEVER imports vector-DB or DB clients directly.

**VectorStore Adapter** — wraps Qdrant. Exposes:
- store_chunks(document_id, chunks_with_vectors) — upsert with dense + sparse vectors
- search(query_vector, sparse_vector, top_k, filters) — hybrid search
- delete_by_document_id(document_id)
- Validates outputs before returning to nodes
- Never retries (nodes own retry logic)

**Postgres Adapter** — wraps all Postgres operations. Exposes:
- Document metadata CRUD
- Changelog CRUD
- Audit writes (append-only)
- Cache reads/writes/flush
- Key lookups for auth
- Validates outputs before returning to nodes
- Never retries (nodes own retry logic)

Swap provider = change one adapter file.

### LLM Adapter (inside adapter-wall)

Wraps LLM calls for Generator node. Each provider implementation:
- Formats structured output request in provider's native way (tool_use for Claude, response_format for OpenAI, prompt enforcement for Llama)
- Returns standardized response shape
- Swap = config change (LLM_PROVIDER + LLM_MODEL_NAME env vars)

---

## Section 4: MCP Architecture

```
Server type:           ICL (internal tooling)
Transport:             HTTP/SSE
System mapping:        Thin wrapper around FastAPI. Zero direct connections
                       to Postgres, Qdrant, or models. MCP calls FastAPI
                       over HTTP only.
Statefulness:          Stateless. No sessions.
Streaming:             No (v2)
```

### Capabilities

**Tool: llm_wiki_query**
- Input: { question: string (required, max 2000 chars), filter: { source_label?: string, document_id?: string } (optional) }
- Output: { answer: string, citations: list, source_chunks: list|null, cached: bool, query_id: string, degraded: bool, session_id: string }
- Calls POST /query on FastAPI

**Resource: source_listing**
- Returns list of available documents
- Calls GET /documents on FastAPI

No write operations exposed via MCP. Writes are admin-only via FastAPI directly.

### Auth
- API key via X-API-Key header, single tier (read-only)
- Key validated per request

### Error Shape
```json
{
  "code": "string",
  "message": "string",
  "retryable": true/false
}
```

Error codes:
| Code | When | Retryable |
|------|------|-----------|
| invalid_input | Empty/malformed question, bad filter | No |
| embedding_error | Embedding model failed | Yes |
| retrieval_error | Qdrant unreachable | Yes |
| insufficient_results | No chunks above 0.65 | No |
| generation_error | LLM failed | Yes |
| grounding_failed | All citations stripped | No |
| listing_error | Postgres unreachable | Yes |

MCP maps FastAPI "error" field to MCP "code" field. Unmapped errors default to "internal_error" retryable:true.

### Security
- Input validation: non-empty question, max 2000 chars, valid filter values
- Output sanitization: never pass raw FastAPI errors to agents, clean error shape only
- Rate limiting: 10 requests/minute per API key (configurable via env var)
- HTTPS in production
- Never log full API keys (prefix only, first 4 chars)
- Timeout: 35 seconds (slightly above query's 30s)

---

## Section 5: FastAPI Implementation

### Database

PostgreSQL, async access. Session-per-request with rollback-on-exception. Alembic for migrations from day one. Two database roles:
- Admin role: migrations only (can ALTER, CREATE, DROP)
- Restricted role: runtime app (can SELECT, INSERT, UPDATE, DELETE only — cannot DROP or TRUNCATE)

### Postgres Tables

**documents**
- document_id (PK, UUID)
- title (varchar 200)
- source_label (varchar 100)
- file_type (varchar 10)
- file_size_bytes (int)
- content_hash (varchar 64, unique — for duplicate detection)
- chunk_count (int)
- ingested_at (timestamp UTC)

**changelog**
- changelog_id (PK, UUID)
- entry (text, max 1000 chars)
- document_id (FK to documents, SET NULL on delete, nullable)
- actor (varchar)
- created_at (timestamp UTC)
- updated_at (timestamp UTC)

**audit_log** (append-only: INSERT only, never UPDATE or DELETE)
- event_id (PK, UUID)
- event_type (varchar)
- document_id (varchar, nullable)
- query_text (text, nullable)
- chunks_retrieved (int, nullable)
- answer_text (text, nullable)
- actor (varchar)
- timestamp (timestamp UTC)
- status (varchar)
- error_detail (text, nullable)
- idempotency_key (unique: thread_id + event_type)

**cache**
- cache_key (PK, varchar 64 — SHA-256 hash)
- question_text (text)
- answer (text)
- citations (JSONB)
- created_at (timestamp UTC)
- expires_at (timestamp UTC — created_at + 48 hours TTL)

**api_keys**
- key_id (PK, UUID)
- key_hash (varchar 64, unique — SHA-256 of the key)
- tier (varchar: "admin" / "service" / "employee")
- actor_name (varchar)
- active (boolean, default true)
- created_at (timestamp UTC)

### Routes (14 endpoints)

| Method | Path | Purpose | Auth Tier |
|--------|------|---------|-----------|
| POST | /documents | Ingest document (multipart) | admin |
| GET | /documents | List all documents (paginated) | admin, service |
| GET | /documents/{id} | Single document metadata | admin, service |
| DELETE | /documents/{id} | Delete document | admin |
| PUT | /documents/{id} | Update document | admin |
| POST | /query | Submit question | admin, service |
| POST | /changelog | Create changelog entry | admin, employee |
| GET | /changelog | List changelog entries (paginated) | admin, employee |
| GET | /changelog/{id} | Single changelog entry | admin, employee |
| PUT | /changelog/{id} | Update changelog entry | admin, employee |
| DELETE | /changelog/{id} | Delete changelog entry | admin, employee |
| GET | /audit | List audit events (paginated) | admin |
| GET | /audit/{id} | Single audit event detail | admin |
| GET | /health | Health check | public (no auth) |

### Request/Response Contracts

**POST /query**
- Request: `{ question: str (required, max 2000), filter: { source_label?: str, document_id?: str } (optional), session_id?: str }`
- Response 200: `{ answer: str, citations: list[Citation], source_chunks: list[RetrievedChunk]|null, cached: bool, query_id: str, degraded: bool, session_id: str }`

**POST /documents**
- Request: multipart — file (required, max 20MB, extensions .pdf/.docx/.txt/.md) + metadata `{ title: str (required, max 200), source_label: str (required, max 100), changelog_id?: str }`
- Response 201: `{ document_id: str, title: str, source_label: str, chunk_count: int, ingested_at: str }`

**GET /documents**
- Response 200: `{ items: list[{ document_id, title, source_label, ingested_at }], next_cursor: str|null }`

**GET /documents/{id}**
- Response 200: `{ document_id, title, source_label, chunk_count, ingested_at, file_type, file_size_bytes }`

**DELETE /documents/{id}**
- Response 204: `{ deleted: true, document_id: str }`

**PUT /documents/{id}**
- Request: same as POST /documents
- Response 200: same as POST /documents response

**POST /changelog**
- Request: `{ entry: str (required, max 1000), document_id?: str }`
- Response 201: `{ changelog_id, entry, document_id, actor, created_at }`

**GET /changelog**
- Response 200: `{ items: list[{ changelog_id, entry, document_id, actor, created_at }], next_cursor: str|null }`

**GET /changelog/{id}**
- Response 200: `{ changelog_id, entry, document_id, actor, created_at, updated_at }`

**PUT /changelog/{id}**
- Request: `{ entry: str (max 1000), document_id?: str }`
- Response 200: same as GET /changelog/{id}

**DELETE /changelog/{id}**
- Response 204: `{ deleted: true, changelog_id: str }`

**GET /audit**
- Response 200: `{ items: list[{ event_id, event_type, document_id, actor, timestamp, status }], next_cursor: str|null }`

**GET /audit/{id}**
- Response 200: `{ event_id, event_type, document_id, query_text, chunks_retrieved, answer_text, actor, timestamp, status, error_detail }`

**GET /health**
- Response 200: `{ status: "healthy"|"degraded", audit_backlog: bool, audit_poisoned: bool, cache_stale_risk: bool, qdrant_connected: bool, postgres_connected: bool, embedding_model_available: bool }`

### Pagination
- Cursor-based on all list endpoints
- Default 20 items per page
- Max 100 items per page (anything above silently capped)

### Auth Flow
1. Request arrives
2. Rate limiter checks per-key limits
3. API key extracted from X-API-Key header
4. key_hash computed, looked up in api_keys table
5. active flag checked (false → 401)
6. tier checked against route permission (wrong tier → 401)
7. All 401s return identical response — no clue about failure type
8. actor_name from key record used for audit trail

### Rate Limits
- Admin keys: 30/minute
- Service keys: 10/minute
- Employee keys: 10/minute
- Health endpoint: unlimited

### Tiered Timeouts
- Query: 30 seconds
- Ingestion: 120 seconds
- Delete/Update: 30 seconds
- Changelog/Audit/Listing: 10 seconds

### Error Semantics

| Code | Meaning | When |
|------|---------|------|
| 200 | Success | Query answered, list returned, update complete |
| 201 | Created | Document ingested, changelog entry created |
| 204 | Deleted | Document or changelog entry deleted |
| 400 | Bad request | Empty question, missing file, invalid filter |
| 401 | Unauthorized | Missing key, invalid key, wrong tier, revoked key |
| 404 | Not found | Document or changelog ID doesn't exist |
| 409 | Conflict | Duplicate document (same content hash) |
| 413 | Too large | File exceeds 20MB |
| 415 | Unsupported type | File not .pdf/.docx/.txt/.md |
| 422 | Validation error | Wrong type, exceeds max length, no extractable text, corrupted file |
| 429 | Rate limited | Too many requests for this key tier |
| 500 | Internal error | Unexpected failure (generic body, no internals) |
| 503 | Service unavailable | Qdrant or Postgres unreachable |

Error body shape (consistent across all errors):
```json
{
  "status": 400,
  "error": "bad_request",
  "message": "Question cannot be empty",
  "retryable": false
}
```

Rules:
- Stacktraces logged server-side with full context, never returned to client
- All 401s return identical body — cannot probe failure type
- 500 body is always generic: "Internal server error." Zero internal detail
- No secrets in server-side logs

### Middleware Order (deterministic)
1. Rate limiter → reject floods before doing any work
2. Auth → reject unauthorized before processing
3. Request logging → structured JSON (method, path, status, duration, actor, key prefix)
4. Endpoint logic

### Cross-Cutting
- CORS: blocked (no browser origins in v1). Frontend + CORS = v2.
- Logging: structured JSON per request. Never log full API keys (first 4 chars only), file contents, embeddings, or raw answers.
- HTTPS: required in production. Dev/demo may use HTTP.
- Config: all from env vars. Fail fast on startup if missing. .env.example ships with code.

### Environment Variables

```
ENV_MODE=production              # dev / demo / production
DATABASE_URL=                    # Postgres connection (restricted role)
DATABASE_ADMIN_URL=              # Postgres connection (migration role)
QDRANT_URL=http://qdrant:6333   # Qdrant connection
QDRANT_COLLECTION=llm_wiki      # Collection name
EMBEDDING_MODEL_NAME=nomic-embed-text-v1.5
LLM_PROVIDER=claude             # claude / openai / llama
LLM_MODEL_NAME=claude-sonnet-4-6
LLM_API_KEY=                    # Claude/OpenAI API key (blank for Llama)
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=1000
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CACHE_TTL_HOURS=48
HYBRID_DENSE_WEIGHT=0.7
HYBRID_SPARSE_WEIGHT=0.3
QUALITY_THRESHOLD=0.65
TOP_K=5
MAX_FILE_SIZE_MB=20
MAX_QUERY_LENGTH=2000
RATE_LIMIT_ADMIN=30
RATE_LIMIT_SERVICE=10
RATE_LIMIT_EMPLOYEE=10
DEAD_LETTER_PATH=/data/dead-letter
APP_VERSION=1.0.0
```

### Cache System
- Postgres table (see schema above)
- Key: normalized hash (lowercase, strip punctuation, collapse whitespace, SHA-256)
- TTL: 48 hours (configurable)
- Invalidated (all rows flushed) on any ingest/delete/update
- Cache flush: 1 retry. If still fails → health flag cache_stale_risk=true
- Stale cache self-heals when TTL expires
- Cache Checker: 0 retries (best-effort, miss = run normal pipeline)
- Cache Writer: 0 retries (best-effort)

### Dead-Letter System
- Persistent volume (host folder surviving container restarts)
- Path from DEAD_LETTER_PATH env var
- Written when Audit Writer fails 3 retries
- Startup replay routine: on app start, read dead-letter file, attempt to write each entry to Postgres
- Retry counter per entry
- After 3 failed replays → move entry to poisoned file in same volume
- Poisoned entries never replayed again
- Admin reviews manually
- Health endpoint flags: audit_backlog (dead-letter has entries), audit_poisoned (poisoned file has entries)

### Key Management
- All keys (admin, service, employee) in one Postgres table
- key_hash stored (SHA-256 of the actual key), never the raw key
- tier: admin / service / employee
- active: boolean (admin sets false to revoke)
- actor_name: who this key belongs to (for audit trail)
- Admin creates keys (mechanism: CLI command or admin endpoint — TBD by implementer)

---

## Section 6: Security Decisions

```
Sensitive data:        Company SOPs, policies, internal procedures.
                       API keys for auth. LLM API keys for generation.

Required blocks:
  - Input validation on every endpoint (Pydantic + file type + size + content check)
  - Output sanitization (no raw errors, no stacktraces to client)
  - API key auth on every route except /health
  - Rate limiting per tier
  - HTTPS in production
  - Never log full API keys, file contents, or query answers
  - Parameterized queries (SQLAlchemy, never raw SQL)
  - Least-privilege DB roles (restricted runtime, admin for migrations)
  - Key revocation via active flag in Postgres

Auth flow:
  Request → rate limiter → extract X-API-Key header → compute key_hash →
  lookup in api_keys table → check active flag → check tier against route →
  proceed or 401 (identical for all failures)

Logging rules:
  - Structured JSON per request (method, path, status, duration, actor)
  - API key prefix only (first 4 chars), never full key
  - Never log file contents, embeddings, or raw answers
  - Server-side full stacktrace on errors, never client-facing

HTTPS:                 Production: required. Dev/demo: HTTP allowed.
CORS:                  Blocked in v1. No browser origins.
```

---

## Section 7: Decision Ledger

### LangGraph + RAG Domain (46 decisions)

D-1: Authority scope — Query autonomous, writes human-gated — SURVIVED
D-2: Operation inventory — 4 ops — FORTIFIED: update order reversed (ingest new first, delete old second)
D-3: Auto-resolve vs escalate — single admin, no queue — SURVIVED
D-4: Changelog — FastAPI CRUD, Postgres table — FORTIFIED: SET NULL on FK when document deleted
D-5: Chunk ranking — similarity first, recency second — SURVIVED
D-6: Entry point — Pydantic-validated JSON — SURVIVED
D-7: Adapter-wall — two adapters — FORTIFIED: adapters never retry, nodes own retries, adapters validate outputs
D-8: Audit trail — append-only — FORTIFIED: dead-letter retry counter, poisoned file after 3 replays, health flag
D-9: Thread definition — one per operation, session_id groups — SURVIVED
D-10: Human SLA — skipped — SURVIVED
D-11: State schema — 19 fields — FORTIFIED: adapters validate outputs before returning to nodes
D-12: RunnableConfig — sealed envelope — SURVIVED
D-13: Serializer safety — base64, ISO-8601 — SURVIVED
D-14: Node inventory — 14 nodes — SURVIVED
D-15: Model assignment — Generator only — SURVIVED
D-16: Routing — 4 points — FORTIFIED: cache flush gets 1 retry, health flag cache_stale_risk, 48hr TTL
D-17: Fan-out — none, v2 — SURVIVED
D-18: HITL — skipped — SURVIVED
D-19: Checkpointer — AsyncPostgresSaver — SURVIVED
D-20: Thread_id lifecycle — generated at boundary — SURVIVED
D-21: Crash behavior — idempotency — FORTIFIED: Storer wipes existing chunks for document_id before writing new set
D-22: Query result caching — SURVIVED (covered by #16 fortification)
D-23: Topology — single graph, four paths — SURVIVED
D-24: Tool inventory + failure modes — FORTIFIED: Storer gets 1 retry, rollback Postgres on Qdrant failure
D-25: Static toolset — SURVIVED
D-26: Retry policy — SURVIVED (updated by #7, #16, #24 fortifications)
D-27: Failure-mode sweep — SURVIVED
D-28: Streaming — none, v2 — SURVIVED
D-29: Multi-tenancy — deferred, v2 — SURVIVED
D-30: RunnableConfig tags — SURVIVED
D-31: Prompt caching — deferred, v2 — SURVIVED
D-32: Deployment — Docker Compose — SURVIVED
D-33: State↔tool shapes — SURVIVED
D-34: Config mode toggle — SURVIVED
D-35: Embedding model — Nomic default, eval compares — FORTIFIED: health endpoint checks embedding model
D-36: Vector DB — Qdrant self-hosted — SURVIVED
D-37: Chunking — 500 tokens, structure-aware — SURVIVED
D-38: Retrieval — top-5, 0.65 threshold — SURVIVED
D-39: Citations — 3-layer grounding — FORTIFIED: structured output for citations, try/catch safety net for parse failures
D-40: Retrieval-vs-generation — heavy retrieval, LLM is formatter — SURVIVED
D-41: Hybrid search — dense+sparse 0.7/0.3 — SURVIVED
D-42: Chunk context enrichment — prepend metadata — SURVIVED
D-43: File types — PDF/DOCX/TXT/MD — FORTIFIED: parser wrapped in try/catch, clean 422 on corrupted/locked files
D-44: Cache key — normalized hash SHA-256 — SURVIVED
D-45: LLM for Generator — Sonnet default, eval compares — SURVIVED
D-46: State cleanup — clear document_file after chunking — SURVIVED

### MCP Domain (8 decisions)

MCP-1: Capability mapping — 1 tool (query), 1 resource (source listing) — SURVIVED
MCP-2: Transport — HTTP/SSE — SURVIVED
MCP-3: Auth — API key, single tier (read-only) — SURVIVED
MCP-4: Tool granularity — one tool, one resource, no writes via MCP — SURVIVED
MCP-5: Failure legibility — 3-field error shape, 7 error codes — SURVIVED
MCP-6: Statefulness — stateless — SURVIVED
MCP-7: System mapping — thin wrapper around FastAPI — FORTIFIED: tiered timeouts per operation
MCP-8: Security — 5 requirements — SURVIVED

### FastAPI Domain (10 decisions)

API-1: Resource model — 14 endpoints, multipart upload, 20MB max, pagination — SURVIVED
API-2: Data contracts — all Pydantic shapes locked — SURVIVED
API-3: Persistence — async Postgres, Alembic, two DB roles — SURVIVED
API-4: Auth model — 3 tiers — FORTIFIED: keys in Postgres table with active flag for revocation
API-5: Concurrency — all async, inline processing — SURVIVED
API-6: Error semantics — 13 status codes, consistent body — SURVIVED
API-7: Cross-cutting — CORS blocked, structured logging, rate limits, middleware order, env vars — SURVIVED
API-7b: Deployment packaging — 4 containers, 3 volumes, docker compose up — SURVIVED
API-8: Test surface — ~52 tests — SURVIVED
API-9: Gap fixes — empty file, degraded response, source_label cap, pagination cap, session_id, tiered timeouts — SURVIVED

### Seam Findings

Seam 1: Double cache flush on update — NO GAP (idempotent)
Seam 2: MCP↔FastAPI error shape — FORTIFIED: MCP maps API "error" to MCP "code" field
Seam 3: Structured output across providers — FORTIFIED: LLM adapter owns translation per provider
Seam 4: State cleanup + crash recovery — NO GAP
Seam 5: Key storage consistency — FORTIFIED: all keys in one Postgres table

---

## Section 8: Success Criteria

### Functional
- Query returns grounded answers with valid citations
- Ingestion parses PDF/DOCX/TXT/MD, chunks, embeds, stores
- Delete removes all chunks and metadata
- Update replaces document without data loss (ingest first, delete second)
- Cache serves repeated questions, invalidates on KB changes
- Changelog CRUD works for employees, admin sees all entries
- Audit trail records every operation
- Source listing returns all documents
- MCP agents query successfully via llm_wiki_query tool

### Security
- No ungrounded citations pass verification
- All 401s identical regardless of failure type
- No stacktraces reach clients
- No full API keys in logs
- Revoked keys immediately rejected
- HTTPS enforced in production

### Performance
- Query under 11 seconds
- Cache hits under 1 second
- Ingestion within 120 seconds for files up to 20MB
- Health endpoint under 2 seconds (including embedding model check)

### Agent-Native
- External agent can query via MCP and receive grounded answer with citations in a single call

---

## Section 9: Build Order

```
Step 1:  Postgres schema — documents, changelog, audit, cache, api_keys tables.
         Alembic migration setup. Two DB roles.
Step 2:  Adapter-wall — VectorStore adapter (Qdrant) + Postgres adapter.
         Output validation on both.
Step 3:  LLM adapter — Claude/OpenAI/Llama structured output translation.
Step 4:  File parsers — PDF, DOCX, TXT, MD. Try/catch per parser.
Step 5:  Chunker node — structure-aware splitting + context enrichment + state cleanup.
Step 6:  Embedding integration — Nomic model loading + Embedding Batcher node.
Step 7:  Qdrant setup — collection with dense + sparse vector support.
Step 8:  LangGraph state schema + graph wiring — all 14 nodes, 4 routing points,
         checkpointer.
Step 9:  Ingestion path — Validator → Duplicate Checker → Chunker → Embedding Batcher
         → Storer → Audit Writer. Including Storer fortifications (#21, #24).
Step 10: Query path — Validator → Cache Checker → Embedder → Retriever → Ranker →
         Quality Gate → Generator → Cache Writer → Audit Writer. Including hybrid
         search, structured output, grounding verification, degraded fallback.
Step 11: Delete/Update path — including fortified update order (#2), FK SET NULL (#4).
Step 12: FastAPI endpoints — all 14 routes, Pydantic request/response models, auth
         middleware (key table lookup), rate limiting per tier, error handlers,
         tiered timeouts, middleware order.
Step 13: Cache system — write, lookup, flush/invalidation, TTL, health flags.
Step 14: Dead-letter system — persistent volume, startup replay, retry counter,
         poisoned file, health flags.
Step 15: Health endpoint — all 7 flags (status, audit_backlog, audit_poisoned,
         cache_stale_risk, qdrant_connected, postgres_connected,
         embedding_model_available).
Step 16: MCP server — thin HTTP wrapper, llm_wiki_query tool, source_listing resource,
         API key auth, error mapping (API error → MCP code), rate limiting, timeout.
Step 17: Docker Compose — 4 containers (App, Postgres, Qdrant, MCP), 3 persistent
         volumes, container networking, .env/.env.example, secrets at runtime.
Step 18: Tests — ~52 tests, in-memory Postgres, mocked Qdrant. 3 per endpoint +
         specific failure tests. Failing test blocks deploy.
Step 19: README.md — system description, Mermaid flowchart (all 14 nodes + routing),
         Mermaid sequence diagram (query flow + error path), setup instructions,
         v2 upgrade paths.
Step 20: Code-debugger validation pass.
```

---

## Section 10: README.md Specification

### 1. System Description
LLM Wiki is a production-grade RAG knowledge base for enterprise SOPs. It ingests company documents (PDF, DOCX, TXT, Markdown), chunks and embeds them, stores them in a vector database, and answers natural-language questions with grounded, cited responses traceable to exact source sections. Managed by a single admin. Queryable by humans and AI agents via MCP. Built with LangGraph, FastAPI, Qdrant, and PostgreSQL.

### 2. Mermaid Flowchart
Must include: all 14 nodes, 4 routing points with conditions, 4 operation paths (query, ingest, delete, update). Show the update path's fortified order (ingest new → verify → delete old).

### 3. Mermaid Sequence Diagram
Must include: full query flow (Agent → MCP → FastAPI → Cache Check → Embed → Retrieve → Rank → Quality Gate → Generate → Cache Write → Audit → Response). Also show: cache hit shortcut, quality gate rejection, degraded fallback (LLM failure), and error path.

### 4. Setup Instructions
- Clone repo
- Copy .env.example to .env, fill in required values
- docker compose up (starts App, Postgres, Qdrant, MCP)
- Run Alembic migrations: alembic upgrade head
- Create admin API key (CLI command or script)
- Upload first document: POST /documents with multipart form
- Query: POST /query with JSON body
- MCP: connect agent to MCP server URL with API key

### 5. V2 Upgrade Paths
Each with exact files/decisions that reopen:
- **Conversational memory (#9):** Add session state to graph, modify retrieval to include prior context, add follow-up prompt chaining. Files: state schema, Retriever node, Generator prompt, API session handling.
- **Batch ingestion (#17):** Add Send() fan-out, partial-failure handling, background task queue. Files: graph routing, Storer node, new queue infrastructure.
- **Streaming (#28):** Switch invoke() to stream_events(), add SSE endpoint. Files: FastAPI query route, MCP tool response.
- **Multi-tenancy (#29):** Add tenant_id to all tables and Qdrant filters, route by tenant. Files: all adapters, all Postgres tables, Qdrant collection strategy, auth model.
- **Prompt caching (#31):** Add after provider and prompt are stable. Files: Generator node, LLM adapter.
- **CSV/Excel/HTML support (#43):** Add parsers to Chunker. Files: Chunker node parser registry.
- **Browser frontend + CORS:** Add frontend domain to CORS config. Files: FastAPI middleware, CORS settings.

---

## Crash Risks (document during eval with cause/symptom/fix/alternative)

1. **Memory — embedding model + all containers on small VPS**
   - Cause: Nomic loads ~500MB-1GB into RAM. Add Postgres, Qdrant, app. 2-4GB VPS is at the edge.
   - Symptom: OOM kill. Container dies mid-operation.
   - Fix: 8GB minimum VPS. Monitor memory usage.
   - Alternative: Swap to smaller embedding model (MiniLM, 384d, ~100MB RAM).

2. **Base64 checkpoint bloat**
   - Cause: document_file as base64 checkpointed at every node.
   - Symptom: Postgres disk fills silently.
   - Fix: Decision #46 — Chunker sets document_file=None after use. Verify in eval.
   - Alternative: Don't checkpoint the ingestion path (loss of crash recovery for ingestion).

3. **Large file parsing — memory spike + timeout**
   - Cause: 500-page PDF loaded fully into memory, parsed, hundreds of chunks embedded on CPU.
   - Symptom: Memory spike, API timeout (120s exceeded), admin sees error.
   - Fix: 20MB file size limit. 120s timeout. Monitor.
   - Alternative: Streaming parser that processes pages one at a time (v2).

4. **Concurrent ingestion + query — embedding model bottleneck**
   - Cause: Both operations need the embedding model. CPU processes sequentially.
   - Symptom: Query latency spikes during ingestion.
   - Fix: Accept for v1 (single admin, low volume). Document the behavior.
   - Alternative: Separate embedding workers or API-based embedding (v2).

---

## Engineer Verification Tasks

1. Verify AsyncPostgresSaver exact class name vs current LangGraph docs (#19)
2. Verify Qdrant sparse vector API for hybrid search (#41)
3. Verify Nomic-embed-text-v1.5 loading method and memory footprint (#35)
4. Verify structured output support per LLM provider (Claude tool_use, OpenAI response_format, Llama)
