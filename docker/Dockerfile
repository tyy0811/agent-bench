FROM python:3.11-slim
WORKDIR /app

# Copy all source before pip install (package needs agent_bench/ to build)
COPY pyproject.toml .
COPY agent_bench/ agent_bench/
COPY configs/ configs/
COPY data/ data/
COPY scripts/ scripts/

RUN pip install --no-cache-dir .

# Pre-download models at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Run ingestion at build time so the store is ready
RUN python scripts/ingest.py --doc-dir data/tech_docs/ --store-path .cache/store

EXPOSE 8000
CMD ["uvicorn", "agent_bench.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
