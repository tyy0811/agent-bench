# Middleware in FastAPI

Middleware is a function that processes every request before it reaches a route handler and every response before it is returned to the client. FastAPI supports both ASGI middleware (from Starlette) and its own decorator-based middleware.

## Custom Middleware

Use the `@app.middleware("http")` decorator to create custom middleware:

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response
```

The middleware receives the incoming `Request` object and a `call_next` function. Calling `await call_next(request)` passes the request to the next middleware or route handler in the chain and returns the `Response`. You can modify both the request (before `call_next`) and the response (after `call_next`).

## CORS Middleware

Cross-Origin Resource Sharing (CORS) is configured using `CORSMiddleware` from Starlette:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com", "https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Custom-Header"],
    max_age=600,
)
```

The `CORSMiddleware` parameters:

| Parameter            | Default | Description                                        |
|----------------------|---------|----------------------------------------------------|
| `allow_origins`      | `[]`    | List of allowed origin URLs                        |
| `allow_origin_regex` | `None`  | Regex pattern for matching allowed origins         |
| `allow_methods`      | `["GET"]` | HTTP methods allowed for cross-origin requests  |
| `allow_headers`      | `[]`    | HTTP headers allowed in cross-origin requests      |
| `allow_credentials`  | `False` | Whether cookies are permitted in cross-origin requests |
| `expose_headers`     | `[]`    | Response headers accessible to the browser         |
| `max_age`            | `600`   | Seconds the browser caches preflight results       |

To allow all origins, use `allow_origins=["*"]`. However, when `allow_credentials=True`, you cannot use the wildcard `"*"` for `allow_origins` -- you must list specific origins. This is a CORS specification requirement, not a FastAPI limitation.

## Middleware Ordering

Middleware executes in reverse order of how it is added. The last middleware added is the first to process the request (outermost layer):

```python
app = FastAPI()

@app.middleware("http")
async def middleware_one(request: Request, call_next):
    print("Middleware 1: before")  # Runs second
    response = await call_next(request)
    print("Middleware 1: after")   # Runs third
    return response

@app.middleware("http")
async def middleware_two(request: Request, call_next):
    print("Middleware 2: before")  # Runs first
    response = await call_next(request)
    print("Middleware 2: after")   # Runs fourth
    return response
```

The output order for a request is: `Middleware 2: before`, `Middleware 1: before`, (route handler), `Middleware 1: after`, `Middleware 2: after`. This follows the standard "onion" model where each middleware wraps the next layer.

## Trusted Host Middleware

Protect against HTTP Host header attacks:

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"],
)
```

Requests with a `Host` header not matching the allowed hosts receive a 400 Bad Request response.

## GZip Middleware

Compress responses automatically when the client supports it:

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=500)
```

The `minimum_size` parameter (default: `500` bytes) sets the minimum response body size before compression is applied. Responses smaller than this threshold are sent uncompressed. GZip compression typically reduces JSON response sizes by 60-80%.

## ASGI Middleware

Since FastAPI is an ASGI application, you can use any ASGI-compatible middleware:

```python
from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    SessionMiddleware,
    secret_key="your-session-secret",
    max_age=14 * 24 * 60 * 60,  # 14 days in seconds = 1,209,600
)
```

The `add_middleware()` method is the preferred way to add middleware in FastAPI, as it ensures proper integration with the application's middleware stack and exception handling.
