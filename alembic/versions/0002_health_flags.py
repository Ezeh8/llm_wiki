"""health_flags table

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-07

Durable operational health flags (e.g. cache_stale_risk — D-16/D-24 fortifications).
The PRD's Section 5 table list doesn't include one; this is Step 9's storage choice
for a behavior the PRD specifies ("set health flag cache_stale_risk=true") without
naming a mechanism — see BUILD_LOG Step 9. Restricted role gets the same
SELECT/INSERT/UPDATE grant as the other mutable tables (no DELETE — flags are cleared
by setting value=false, not by deleting rows).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import make_url

from app.config import get_settings

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Migrations must stay self-contained (never import another migration module — its
# name isn't even a valid Python identifier, and cross-migration imports would break
# if 0001 were ever edited). Same small helpers as 0001, duplicated on purpose.
def _runtime_role() -> tuple[str, str, str]:
    url = make_url(get_settings().database_url)
    if not url.username or url.password is None:
        raise RuntimeError("DATABASE_URL must include the restricted role username and password")
    if '"' in url.username:
        raise RuntimeError("restricted role name must not contain double quotes")
    return url.username, url.password, url.database


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def upgrade() -> None:
    op.create_table(
        "health_flags",
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("value", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("name", name="pk_health_flags"),
    )
    role, _password, _database = _runtime_role()
    role_ident = _quote_ident(role)
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON health_flags TO {role_ident}")


def downgrade() -> None:
    role, _password, _database = _runtime_role()
    role_ident = _quote_ident(role)
    op.execute(f"REVOKE ALL ON health_flags FROM {role_ident}")
    op.drop_table("health_flags")
