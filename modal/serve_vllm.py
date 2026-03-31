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

# Inlined from common.py — Modal containers don't auto-include sibling modules
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
VLLM_MAX_MODEL_LEN = 8192
VLLM_DTYPE = "half"
VLLM_GPU_MEMORY_UTILIZATION = 0.85

MODELS_DIR = "/models"
VLLM_PORT = 8000
VLLM_READY_TIMEOUT = 600  # seconds to wait for vLLM to become ready (download + load)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.6.post1",
        "transformers==4.47.0",
        "huggingface_hub[hf_transfer]<1.0",
        "httpx",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("agent-bench-vllm")
model_volume = modal.Volume.from_name("vllm-model-cache", create_if_missing=True)


@app.function(
    image=vllm_image,
    gpu="a10g",
    scaledown_window=600,
    timeout=900,
    volumes={MODELS_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.asgi_app()
def serve():
    """Serve vLLM with OpenAI-compatible API.

    Exposes /v1/chat/completions and /health.
    Waits for the vLLM subprocess to be ready before accepting requests.
    """
    import subprocess
    import time

    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    vllm_process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--download-dir", MODELS_DIR,
            "--dtype", VLLM_DTYPE,
            "--max-model-len", str(VLLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION),
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
        ],
    )

    # Wait for vLLM to be ready before accepting proxied requests
    base = f"http://localhost:{VLLM_PORT}"
    deadline = time.monotonic() + VLLM_READY_TIMEOUT
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=2.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if vllm_process.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited with code {vllm_process.returncode}"
            )
        time.sleep(2)
    else:
        vllm_process.terminate()
        raise TimeoutError(
            f"vLLM did not become ready within {VLLM_READY_TIMEOUT}s"
        )

    proxy_app = FastAPI()
    client = httpx.AsyncClient(base_url=base, timeout=120.0)

    @proxy_app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        """Proxy all requests to the vLLM subprocess."""
        import traceback as _tb
        try:
            return await _proxy_inner(path, request)
        except Exception as exc:
            _tb.print_exc()
            return JSONResponse(
                content={"error": str(exc), "type": type(exc).__name__},
                status_code=502,
            )

    async def _proxy_inner(path: str, request: Request):
        url = f"/{path}"
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        # Detect streaming: check body for "stream": true (httpx sends
        # Accept: */*, not text/event-stream, so header check is unreliable)
        is_streaming = False
        if body:
            try:
                import json as _json
                is_streaming = _json.loads(body).get("stream", False)
            except (ValueError, AttributeError):
                pass
        if not is_streaming:
            is_streaming = request.headers.get("accept") == "text/event-stream"

        if is_streaming:
            req = client.build_request(
                request.method, url, content=body, headers=headers
            )
            upstream = await client.send(req, stream=True)

            if upstream.status_code != 200:
                error_body = await upstream.aread()
                await upstream.aclose()
                return Response(
                    content=error_body,
                    status_code=upstream.status_code,
                    media_type="application/json",
                )

            async def stream():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()

            return StreamingResponse(
                stream(),
                status_code=upstream.status_code,
                media_type="text/event-stream",
            )

        resp = await client.request(
            request.method, url, content=body, headers=headers
        )
        # Not all endpoints return JSON (e.g. /health returns empty 200)
        try:
            content = resp.json()
        except Exception:
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "text/plain"),
            )
        return JSONResponse(
            content=content,
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
