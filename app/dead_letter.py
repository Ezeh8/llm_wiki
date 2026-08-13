"""Dead-letter system (PRD Section 5; Node 14 Audit Writer's failure path).

By the time Audit Writer runs, the real operation (ingest/query/delete/update) has
already completed — losing its own audit LOG entry must never retroactively fail the
user's already-finished request. So when all 3 Postgres retries are exhausted, the
event is captured here instead of raised.

Two JSON Lines files on a persistent volume (`DEAD_LETTER_PATH`, a host folder that
survives container restarts — Step 17 wires the actual volume):
  - `dead_letter.jsonl`: entries awaiting replay, each carrying a `retry_count`.
  - `poisoned.jsonl`: entries that failed `MAX_REPLAY_ATTEMPTS` replay attempts — never
    replayed again; an admin reviews this file manually.

`replay_dead_letters()` is the startup replay routine (PRD: "on app start, read
dead-letter file, attempt to write each entry to Postgres") — called once from
FastAPI's lifespan (app.py), not on a recurring schedule.

Single-process, best-effort file I/O (no lock file): concurrent Audit Writer failures
appending to the same file rely on POSIX append-mode writes being effectively atomic
for single lines at this volume — acceptable for how rare a 3x-retry-exhausted audit
write should be. Replay itself only ever runs once, at startup, before the app serves
any traffic, so it never races with a live append.
"""

import json
from pathlib import Path

from app.adapters.postgres import PostgresAdapter
from app.config import get_settings
from app.db.session import async_session_factory

DEAD_LETTER_FILENAME = "dead_letter.jsonl"
POISONED_FILENAME = "poisoned.jsonl"
MAX_REPLAY_ATTEMPTS = 3

_ENTRY_META_KEYS = ("retry_count", "dead_lettered_at")


def _dead_letter_path() -> Path:
    return Path(get_settings().dead_letter_path) / DEAD_LETTER_FILENAME


def _poisoned_path() -> Path:
    return Path(get_settings().dead_letter_path) / POISONED_FILENAME


def _read_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _write_entries(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def write_dead_letter(audit_fields: dict) -> None:
    """Called from the Audit Writer node on retry exhaustion. `audit_fields` must
    already be JSON-serializable (e.g. `timestamp` as an ISO string, not a datetime).
    Best-effort: if even this write fails (e.g. an unwritable volume), there is
    nothing further to do at this layer — the node logs and moves on rather than
    failing the user's already-completed operation.
    """
    path = _dead_letter_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {**audit_fields, "retry_count": 0}
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def has_backlog() -> bool:
    """Health endpoint's `audit_backlog` flag."""
    path = _dead_letter_path()
    return path.exists() and path.stat().st_size > 0


def has_poisoned() -> bool:
    """Health endpoint's `audit_poisoned` flag."""
    path = _poisoned_path()
    return path.exists() and path.stat().st_size > 0


async def replay_dead_letters() -> None:
    """Startup replay: attempt every pending entry once. A successful replay drops
    the entry; a failure increments its retry_count, moving it to the poisoned file
    once MAX_REPLAY_ATTEMPTS is reached. write_audit's own idempotency (unique
    idempotency_key, ON CONFLICT DO NOTHING) makes a replay of an event that actually
    DID make it to Postgres on a prior attempt (but whose dead-letter write also raced
    in) safe to retry — it just no-ops instead of duplicating.
    """
    entries = _read_entries(_dead_letter_path())
    if not entries:
        return

    still_pending: list[dict] = []
    poisoned: list[dict] = _read_entries(_poisoned_path())

    for entry in entries:
        fields = {k: v for k, v in entry.items() if k not in _ENTRY_META_KEYS}
        if fields.get("timestamp"):
            from datetime import datetime

            fields["timestamp"] = datetime.fromisoformat(fields["timestamp"])

        try:
            async with async_session_factory() as session:
                await PostgresAdapter(session).write_audit(**fields)
                await session.commit()
            continue  # replayed successfully — drop it, don't carry it forward
        except Exception:
            pass

        entry["retry_count"] = entry.get("retry_count", 0) + 1
        if entry["retry_count"] >= MAX_REPLAY_ATTEMPTS:
            poisoned.append(entry)
        else:
            still_pending.append(entry)

    _write_entries(_dead_letter_path(), still_pending)
    _write_entries(_poisoned_path(), poisoned)
