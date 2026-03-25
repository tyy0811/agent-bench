"""Configuration loading from YAML files via Pydantic models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

# --- Nested config models ---


class AgentConfig(BaseModel):
    max_iterations: int = 3
    temperature: float = 0.0


class ModelPricing(BaseModel):
    input_cost_per_mtok: float
    output_cost_per_mtok: float


class ProviderConfig(BaseModel):
    default: str = "openai"
    models: dict[str, ModelPricing] = {}


class ChunkingConfig(BaseModel):
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64


class RetrievalConfig(BaseModel):
    strategy: str = "hybrid"
    rrf_k: int = 60
    candidates_per_system: int = 10
    top_k: int = 5


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5  # independent of retrieval.top_k


class RAGConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    store_path: str = ".cache/store"
    refusal_threshold: float = 0.0  # 0.0 = disabled (V1 behavior)


class RetryConfig(BaseModel):
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 8.0  # cap for exponential backoff


class EmbeddingConfig(BaseModel):
    model: str = "all-MiniLM-L6-v2"
    cache_dir: str = ".cache/embeddings"


class ServingConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    request_timeout_seconds: int = 30
    rate_limit_rpm: int = 10  # requests per minute per IP


class MemoryConfig(BaseModel):
    enabled: bool = True
    db_path: str = "data/conversations.db"
    max_turns: int = 10


class EvaluationConfig(BaseModel):
    judge_provider: str = "openai"
    golden_dataset: str = "agent_bench/evaluation/datasets/tech_docs_golden.json"


class AppConfig(BaseModel):
    agent: AgentConfig = AgentConfig()
    provider: ProviderConfig = ProviderConfig()
    rag: RAGConfig = RAGConfig()
    retry: RetryConfig = RetryConfig()
    memory: MemoryConfig = MemoryConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    serving: ServingConfig = ServingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


# --- Task config ---


class TaskConfig(BaseModel):
    name: str
    description: str
    system_prompt: str
    document_dir: str = "data/tech_docs/"


class TaskFileConfig(BaseModel):
    task: TaskConfig


# --- Loaders ---


def _resolve_config_dir() -> Path:
    """Resolve configs directory: cwd first, then package-relative fallback."""
    cwd_configs = Path.cwd() / "configs"
    if cwd_configs.is_dir():
        return cwd_configs
    # Fallback: relative to package location (works for installed packages)
    pkg_configs = Path(__file__).resolve().parent.parent.parent / "configs"
    if pkg_configs.is_dir():
        return pkg_configs
    return cwd_configs  # Let the caller get a clear FileNotFoundError


def load_config(path: Path | None = None) -> AppConfig:
    """Load application config from YAML.

    If AGENT_BENCH_ENV is set (e.g. 'production'), loads configs/{env}.yaml
    if it exists, otherwise falls back to default.yaml.
    """
    if path is None:
        import os

        env = os.environ.get("AGENT_BENCH_ENV", "")
        config_dir = _resolve_config_dir()
        env_path = config_dir / f"{env}.yaml"
        path = env_path if env and env_path.exists() else config_dir / "default.yaml"
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(data)


def load_task_config(task_name: str, path: Path | None = None) -> TaskConfig:
    """Load a task-specific config from YAML."""
    if path is None:
        path = _resolve_config_dir() / "tasks" / f"{task_name}.yaml"
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return TaskFileConfig.model_validate(data).task
