# Main app image (FastAPI + LangGraph). Heavy — sentence-transformers/torch for the
# in-process Nomic embedding model (Crash Risk #1). The MCP server has its own,
# deliberately separate and much lighter image (Dockerfile.mcp) — see its header for
# why it must not share this one.

FROM python:3.12-slim

WORKDIR /app

# PyMuPDF/sentence-transformers/torch need a compiler toolchain for some transitive
# deps on slim base images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

# CPU-only torch FIRST, from PyTorch's own CPU wheel index — installed before the rest
# of the app's dependencies so pip sees it already satisfied and never reaches for the
# default PyPI wheel. That default wheel bundles the full NVIDIA CUDA toolkit as
# transitive dependencies (~2-3GB: cuDNN, cuBLAS, cuSPARSE, NCCL, Triton, ...) — a real
# problem discovered while actually building this image, not a hypothetical: nothing
# in this app runs on GPU (Nomic is an in-process CPU model, Crash Risk #1/#4), so that
# entire toolkit would sit in the image completely unused, multiplying both build time
# and image size for zero benefit.
RUN pip install --no-cache-dir --timeout 180 --retries 5 \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --timeout 180 --retries 5 -e .

# Pre-cache tiktoken's cl100k_base vocab at build time (BUILD_LOG Step 5: the Chunker
# would otherwise fetch it over the network on first use in every fresh container).
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Nomic model weights (~500MB-1GB, Crash Risk #1) are deliberately NOT baked in here:
# doing so would make every image build download that much data, which isn't
# appropriate to force unconditionally. To pre-cache them for a faster first query in
# production, add (after the tiktoken RUN line, before ENV below):
#   RUN python -c "from sentence_transformers import SentenceTransformer; \
#       SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"
# Without it, the model lazy-loads on the first real query (Step 6's existing,
# already-tested behavior) — slower first request, no different otherwise.

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
