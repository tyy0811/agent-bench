FROM python:3.11-slim

# HF Spaces requires user ID 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy source and install
COPY --chown=user pyproject.toml .
COPY --chown=user agent_bench/ agent_bench/
COPY --chown=user configs/ configs/
COPY --chown=user data/ data/
COPY --chown=user scripts/ scripts/

RUN pip install --no-cache-dir .

# Pre-download models at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Run ingestion at build time so both corpora stores are ready
RUN python scripts/ingest.py --doc-dir data/tech_docs/ --store-path .cache/store
RUN python scripts/ingest.py --doc-dir data/k8s_docs/ --store-path .cache/store_k8s

# Give user 1000 ownership of build-time artifacts
RUN chown -R user:user .cache/

USER user
EXPOSE 7860
CMD ["uvicorn", "agent_bench.serving.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
