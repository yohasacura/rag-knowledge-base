# ─────────────────────────────────────────────────────────────
# Dockerfile for rag-knowledge-base
#
# Multi-stage build:
#   Stage 1 — builder: installs deps + builds the wheel
#   Stage 2 — runtime: lean image with only what's needed
#
# Build:
#   docker build -t rag-kb .
#
# Run MCP server (stdio):
#   docker run --rm -it -v /path/to/docs:/data rag-kb serve
#
# Run MCP server (HTTP):
#   docker run --rm -p 8080:8080 -v /path/to/docs:/data rag-kb serve --http --host 0.0.0.0
#
# Interactive CLI:
#   docker run --rm -it -v /path/to/docs:/data rag-kb --help
# ─────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build wheel
RUN pip install --no-cache-dir build \
    && python -m build --wheel --outdir /build/dist


# ── Stage 2: runtime ─────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="RAG Knowledge Base Contributors"
LABEL description="100% offline RAG storage and MCP server"

# Runtime system deps (for lxml, Pillow, rapidocr)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxml2 \
        libxslt1.1 \
        libjpeg62-turbo \
        libpng16-16 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the wheel built in stage 1
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Pre-download default models so the image is fully self-contained
RUN python -c "from rag_kb.models import download_models; download_models()"

# Persistent data directory for RAG databases & config
ENV RAG_KB_DATA_DIR=/data
VOLUME ["/data"]

# Default: run the CLI
ENTRYPOINT ["rag-kb"]
CMD ["--help"]
