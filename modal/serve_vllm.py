"""Deploy vLLM on Modal as an OpenAI-compatible endpoint.

Usage:
    modal deploy modal/serve_vllm.py     # Deploy (stays running, prints URL)
    modal serve modal/serve_vllm.py      # Dev mode (auto-redeploys on change)

The printed URL is the MODAL_VLLM_URL for SelfHostedProvider:
    export MODAL_VLLM_URL=https://<your-workspace>--agent-bench-vllm-serve.modal.run/v1
"""

import modal

from common import (
    MODEL_NAME,
    VLLM_DTYPE,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_MAX_MODEL_LEN,
)

MODELS_DIR = "/models"

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("agent-bench-vllm")
model_volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)


@app.cls(
    image=vllm_image,
    gpu=modal.gpu.A10G(),
    container_idle_timeout=300,
    timeout=600,
    volumes={MODELS_DIR: model_volume},
    allow_concurrent_inputs=10,
)
class VLLMServer:
    @modal.enter()
    def start_engine(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        args = AsyncEngineArgs(
            model=MODEL_NAME,
            download_dir=MODELS_DIR,
            dtype=VLLM_DTYPE,
            max_model_len=VLLM_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
        self.engine = AsyncLLMEngine.from_engine_args(args)

    @modal.asgi_app()
    def serve(self):
        from vllm.entrypoints.openai.api_server import (
            build_async_engine_client_and_server,
        )

        # vLLM's OpenAI-compatible server exposes /v1/chat/completions and /health
        _, _, app = build_async_engine_client_and_server(
            model=MODEL_NAME,
            download_dir=MODELS_DIR,
            dtype=VLLM_DTYPE,
            max_model_len=VLLM_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        )
        return app
