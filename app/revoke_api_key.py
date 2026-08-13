"""Admin API key revocation (Step 20 finding: the PRD's Section 8 Success Criteria
promises "revoked keys immediately rejected," and the `active` flag + auth check both
existed, but no code path ever set `active=False` — only `create_api_key.py` existed.
Mirrors that script's shape exactly: a CLI, not an HTTP endpoint, for the same
bootstrapping-independence reason.

Usage: python -m app.revoke_api_key --key-id <uuid>
key_id is printed by `create_api_key.py` at creation time — the raw key itself can't
be looked up again (only its hash is stored), so key_id is the only handle available.
"""

import argparse
import asyncio

from app.adapters.postgres import PostgresAdapter
from app.db.session import async_session_factory


async def revoke_key(*, key_id: str) -> bool:
    async with async_session_factory() as session:
        api_key = await PostgresAdapter(session).set_api_key_active(key_id, active=False)
        if api_key is None:
            return False
        await session.commit()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Revoke an LLM Wiki API key")
    parser.add_argument("--key-id", required=True, help="key_id printed at creation time")
    args = parser.parse_args()

    found = asyncio.run(revoke_key(key_id=args.key_id))
    if found:
        print(f"Key {args.key_id} revoked. It will be rejected on its next use.")
    else:
        print(f"No key found with key_id {args.key_id!r}.")


if __name__ == "__main__":
    main()
