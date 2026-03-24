# Deploying FastAPI Applications

FastAPI applications are deployed using ASGI servers. This guide covers production deployment with Uvicorn, Gunicorn, Docker, and related infrastructure considerations.

## Uvicorn (Single Process)

Uvicorn is the recommended ASGI server for FastAPI. For development:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

For production with a single process:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

Key Uvicorn configuration options:

| Flag              | Default       | Description                              |
|-------------------|---------------|------------------------------------------|
| `--host`          | `127.0.0.1`   | Bind address                             |
| `--port`          | `8000`         | Bind port                                |
| `--workers`       | `1`            | Number of worker processes               |
| `--loop`          | `auto`         | Event loop (auto, asyncio, uvloop)       |
| `--http`          | `auto`         | HTTP protocol (auto, h11, httptools)     |
| `--ws`            | `auto`         | WebSocket protocol (auto, websockets, wsproto) |
| `--log-level`     | `info`         | Logging level (critical, error, warning, info, debug, trace) |
| `--access-log`    | `True`         | Enable/disable access log                |
| `--ws-max-size`   | `16777216`     | Max WebSocket message size (16 MB)       |
| `--timeout-keep-alive` | `5`       | Keep-alive timeout in seconds            |

Using `uvloop` and `httptools` (installed automatically on Linux) provides a 20-30% performance boost over the pure-Python `asyncio` and `h11` alternatives.

## Gunicorn with Uvicorn Workers

For production deployments requiring multiple worker processes, use Gunicorn as the process manager with Uvicorn workers:

```bash
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile -
```

The recommended number of workers is `(2 * CPU_CORES) + 1`. For a server with 4 CPU cores, that is 9 workers. The `--max-requests 1000` flag restarts each worker after handling 1,000 requests, preventing memory leaks. The `--max-requests-jitter 50` adds a random offset (0-50) so workers do not all restart simultaneously.

The `--timeout 120` flag sets the maximum time (in seconds) a worker can take to handle a request before being killed and restarted. The default is 30 seconds. The `--graceful-timeout 30` gives workers 30 seconds to finish current requests during shutdown.

## Docker Deployment

A production-ready Dockerfile:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

Build and run:

```bash
docker build -t myapi:latest .
docker run -d --name myapi -p 8000:8000 -e DATABASE_URL=postgresql://... myapi:latest
```

The `python:3.12-slim` base image is approximately 120 MB, compared to the full `python:3.12` image at approximately 890 MB. For even smaller images, use `python:3.12-alpine` (approximately 50 MB), though it may require additional build dependencies for some Python packages.

## Proxy Headers and HTTPS

When running behind a reverse proxy (Nginx, Traefik, AWS ALB), configure Uvicorn to trust proxy headers:

```bash
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --proxy-headers \
    --forwarded-allow-ips="*"
```

The `--proxy-headers` flag tells Uvicorn to read `X-Forwarded-For` and `X-Forwarded-Proto` headers from the proxy. The `--forwarded-allow-ips` flag specifies which proxy IPs are trusted. Using `"*"` trusts all proxies (acceptable when the application is not directly exposed to the internet).

An Nginx reverse proxy configuration:

```nginx
upstream fastapi_backend {
    server 127.0.0.1:8000;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/api.example.com.pem;
    ssl_certificate_key /etc/ssl/private/api.example.com.key;

    location / {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
    }
}
```

Setting `proxy_buffering off` ensures streamed responses (like SSE or large file downloads) are forwarded immediately rather than buffered by Nginx.

## Health Checks

Include a health check endpoint for container orchestrators:

```python
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "healthy"}
```

Docker health check configuration:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1
```

This checks health every 30 seconds, allows 10 seconds per check, retries 3 times before marking unhealthy, and waits 10 seconds after container start before the first check.
