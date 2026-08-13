"""Qdrant collection provisioning (Step 7).

Stands up the QDRANT_COLLECTION with the exact schema the Step 2 VectorStore adapter
assumes — names/dim imported from that module so the two can never drift:
  - dense  "text"  : size 768, Cosine (Nomic)
  - sparse "bm25"  : native sparse vector with Modifier.IDF (server computes the IDF
                     component of BM25; fastembed supplies TF for docs, 1.0 for queries)

Idempotent: safe to call on every startup or via `python -m app.qdrant_setup`.
"""

import asyncio

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from app.adapters.vector_store import DENSE_VECTOR_NAME, EMBEDDING_DIM, SPARSE_VECTOR_NAME
from app.config import get_settings


async def ensure_collection(
    client: AsyncQdrantClient | None = None, *, recreate: bool = False
) -> bool:
    """Create the collection if absent. Returns True if created, False if it already existed.

    `recreate=True` drops and rebuilds it (admin reset — destroys stored vectors).
    """
    settings = get_settings()
    owns_client = client is None
    client = client or AsyncQdrantClient(url=settings.qdrant_url)
    name = settings.qdrant_collection

    try:
        exists = await client.collection_exists(name)
        if exists and recreate:
            await client.delete_collection(name)
            exists = False
        if exists:
            return False

        await client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: qm.VectorParams(
                    size=EMBEDDING_DIM, distance=qm.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)
            },
        )
        return True
    finally:
        if owns_client:
            await client.close()


def main() -> None:
    created = asyncio.run(ensure_collection())
    print(f"collection {'created' if created else 'already exists'}")


if __name__ == "__main__":
    main()
