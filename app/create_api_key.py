"""Admin API key creation (PRD Section 5: "Admin creates keys (mechanism: CLI command
or admin endpoint — TBD by implementer)"). Implemented as a CLI, not an HTTP endpoint:
an endpoint would need its own admin key to authenticate the request that creates the
first admin key — a bootstrapping problem a CLI run against the database directly
avoids entirely.

Usage: python -m app.create_api_key --tier admin --actor "Jane Admin"
Prints the raw key ONCE — only its SHA-256 hash is stored (Section 6), so this is the
only time it's ever visible. Also prints key_id, needed to revoke this key later via
`python -m app.revoke_api_key` (Step 20: the raw key can't be looked up again — only
its hash is stored — so key_id is the only handle an admin has for revocation).
"""

import argparse
import asyncio
import secrets

from app.adapters.postgres import PostgresAdapter
from app.api.auth import hash_key
from app.db.session import async_session_factory

_TIERS = ("admin", "service", "employee")


async def create_key(*, tier: str, actor_name: str) -> tuple[str, str]:
    if tier not in _TIERS:
        raise ValueError(f"tier must be one of {_TIERS}, got {tier!r}")
    raw_key = secrets.token_urlsafe(32)
    async with async_session_factory() as session:
        api_key = await PostgresAdapter(session).create_api_key(
            key_hash=hash_key(raw_key), tier=tier, actor_name=actor_name
        )
        await session.commit()
        key_id = str(api_key.key_id)
    return raw_key, key_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an LLM Wiki API key")
    parser.add_argument("--tier", required=True, choices=_TIERS)
    parser.add_argument("--actor", required=True, help="actor_name for the audit trail")
    args = parser.parse_args()

    raw_key, key_id = asyncio.run(create_key(tier=args.tier, actor_name=args.actor))
    print(f"API key created for {args.actor!r} ({args.tier}). Save it now — shown once:")
    print(raw_key)
    print(f"key_id (save this too — needed to revoke the key later): {key_id}")


if __name__ == "__main__":
    main()
