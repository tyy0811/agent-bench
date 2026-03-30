"""Deploy vLLM on Modal as an OpenAI-compatible endpoint.

Usage:
    modal deploy modal/serve_vllm.py     # Deploy (stays running, prints URL)
    modal serve modal/serve_vllm.py      # Dev mode (auto-redeploys on change)

The printed URL is the MODAL_VLLM_URL for SelfHostedProvider:
    export MODAL_VLLM_URL=https://<your-workspace>--agent-bench-vllm-serve.modal.run/v1

Note: The vLLM server integration pattern changes between vLLM releases.
      If deployment fails, check Modal's vLLM example for the current API:
      https://modal.com/docs/examples/vllm_inference
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
    .pip_install("vllm>=0.6.0", "huggingface_hub[hf_transfer]", "httpx")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("agent-bench-vllm")
model_volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(),
    container_idle_timeout=300,
    timeout=600,
    volumes={MODELS_DIR: model_volume},
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def serve():
    """Serve vLLM with OpenAI-compatible API.

    Exposes /v1/chat/completions and /health.
    """
    import subprocess

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import httpx

    vllm_process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--download-dir", MODELS_DIR,
            "--dtype", VLLM_DTYPE,
            "--max-model-len", str(VLLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
    )

    proxy_app = FastAPI()
    client = httpx.AsyncClient(base_url="http://localhost:8000", timeout=120.0)

    @proxy_app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        """Proxy all requests to the vLLM subprocess."""
        url = f"/{path}"
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        if request.headers.get("accept") == "text/event-stream":
            async def stream():
                async with client.stream(
                    request.method, url, content=body, headers=headers
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            # For streaming, status is always 200 if the connection opened;
            # vLLM errors surface as SSE error events in the stream body.
            return StreamingResponse(stream(), media_type="text/event-stream")

        resp = await client.request(
            request.method, url, content=body, headers=headers
        )
        return JSONResponse(
            content=resp.json(),
            status_code=resp.status_code,
            headers={
                k: v for k, v in resp.headers.items()
                if k.lower() not in ("content-length", "transfer-encoding")
            },
        )

    @proxy_app.on_event("shutdown")
    def shutdown():
        vllm_process.terminate()

    return proxy_app
