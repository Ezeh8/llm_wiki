"""Stage 2 Step 4 — adversarial case.

Proves the cache-scope-blindness fix (Stage 2 Step 4: normalize_cache_key() now folds
document_metadata.filter into the cache key, app/graph/nodes/query_path.py) actually
closes the gap: before the fix, a finance-scoped question and an identically-worded
legal-scoped question shared one cache entry, so whichever filter answered first got
served to the other. Reuses test_endpoints.py's api_client/keys/_ingest fixtures
verbatim, same pattern as tests/test_golden_dataset.py.
"""

import re
import uuid

from tests.test_endpoints import _ingest, api_client, keys


class _ContentEchoingLLM:
    """Unlike EchoingLLM (test_endpoints.py, which always returns one fixed canned
    answer regardless of what was retrieved), this fake embeds the actual retrieved
    chunk_text into its answer. Needed here specifically: proving cache-scope
    isolation requires two DIFFERENT, content-derived answers to compare — a fixed
    canned answer would make finance- and legal-scoped responses indistinguishable
    even if retrieval/caching were both working perfectly."""

    async def generate(self, *, system, user):
        from app.domain import Citation, GenerationResult

        chunk_id_match = re.search(r"chunk_id=(\S+)", user)
        chunk_id = chunk_id_match.group(1) if chunk_id_match else "unknown"
        doc_match = re.search(r"document='([^']*)'", user)
        # Chunk body text lives after the "[chunk_id=... chunk_index=N]\n" header.
        body_match = re.search(r"\]\n(.+)", user, re.DOTALL)
        body = body_match.group(1).strip() if body_match else ""

        return GenerationResult(
            answer=f"Based on the retrieved document: {body}",
            citations=[
                Citation(
                    document_id="doc",
                    document_title=doc_match.group(1) if doc_match else "Doc",
                    chunk_id=chunk_id,
                    chunk_text=body,
                    chunk_index=0,
                )
            ],
        )


def test_adversarial_cache_scope_isolation(api_client, keys, monkeypatch):
    import app.graph.nodes.query_path as query_mod

    monkeypatch.setattr(query_mod, "build_llm_adapter", lambda: _ContentEchoingLLM())

    finance_fact = f"The reimbursement cap is $500 per month ({uuid.uuid4()})."
    legal_fact = f"The legal notice period for termination is 90 days ({uuid.uuid4()})."

    _ingest(
        api_client,
        keys["admin"],
        filename="finance.md",
        content=f"# Finance Policy\n\n{finance_fact}".encode(),
        title="Finance Policy",
        source_label="finance",
    )
    _ingest(
        api_client,
        keys["admin"],
        filename="legal.md",
        content=f"# Legal Policy\n\n{legal_fact}".encode(),
        title="Legal Policy",
        source_label="legal",
    )

    question = f"What is the policy limit, golden-adversarial edition {uuid.uuid4()}?"

    finance_response = api_client.post(
        "/query",
        headers=keys["service"],
        json={"question": question, "filter": {"source_label": "finance"}},
    )
    legal_response = api_client.post(
        "/query",
        headers=keys["service"],
        json={"question": question, "filter": {"source_label": "legal"}},
    )

    assert finance_response.status_code == 200
    assert legal_response.status_code == 200

    finance_body = finance_response.json()
    legal_body = legal_response.json()

    # Each response must reflect ITS OWN scope's fact, and never the other scope's —
    # the exact leak the cache-scope fix closes.
    assert finance_fact in finance_body["answer"]
    assert legal_fact not in finance_body["answer"]

    assert legal_fact in legal_body["answer"]
    assert finance_fact not in legal_body["answer"]

    # Each is a first-time call for its own (question, filter) pair — neither should
    # be served from a stale/cross-scope cache entry.
    assert finance_body["cached"] is False
    assert legal_body["cached"] is False


def test_adversarial_cache_scope_hit_on_true_repeat(api_client, keys):
    """The fix must not break the legitimate case: identical question AND identical
    filter, asked twice, should still hit the cache the second time."""
    _ingest(
        api_client,
        keys["admin"],
        content=f"# Repeat Scope Doc\n\nThe on-call rotation is 2 weeks ({uuid.uuid4()}).".encode(),
        title="Repeat Scope Doc",
        source_label="ops",
    )

    question = f"What is the on-call rotation, adversarial repeat check {uuid.uuid4()}?"
    payload = {"question": question, "filter": {"source_label": "ops"}}

    first = api_client.post("/query", headers=keys["service"], json=payload)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = api_client.post("/query", headers=keys["service"], json=payload)
    assert second.status_code == 200
    body = second.json()
    assert body["cached"] is True
    assert body["answer"] == first.json()["answer"]
