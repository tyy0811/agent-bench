# Introduction to FastAPI

FastAPI is a modern, high-performance web framework for building APIs with Python 3.7+ based on standard Python type hints. Created by Sebastian Ramirez and first released in December 2018, it has quickly become one of the most popular Python web frameworks, with over 75,000 stars on GitHub.

## Key Features

FastAPI is built on top of two core libraries:

- **Starlette** (version 0.27.0+) for the web framework internals, providing WebSocket support, ASGI compatibility, and background tasks.
- **Pydantic** (version 2.0+) for data validation, serialization, and settings management using Python type annotations.

The framework delivers several standout capabilities:

1. **High Performance**: FastAPI achieves performance on par with Node.js and Go frameworks. Independent benchmarks from TechEmpower show it handling approximately 9,000 requests per second for JSON serialization on a single worker, compared to Flask's approximately 1,200 requests per second under comparable conditions.

2. **Automatic Interactive Documentation**: Every FastAPI application automatically generates two interactive API documentation interfaces -- Swagger UI (available at `/docs`) and ReDoc (available at `/redoc`) -- with zero additional configuration.

3. **Async Support**: Full native support for `async`/`await` syntax, allowing non-blocking I/O operations. Synchronous route handlers are automatically run in a threadpool with a default thread count of 40.

4. **Type-Driven Development**: Leverages Python type hints for request validation, serialization, and documentation generation, reducing code duplication by an estimated 40% compared to traditional approaches.

## Minimal Example

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="A sample API built with FastAPI",
    version="1.0.0",
)

@app.get("/")
async def root():
    return {"message": "Hello, World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
```

To run this application, save it as `main.py` and execute:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes and should only be used during development. By default, Uvicorn binds to `127.0.0.1` on port `8000`.

## How It Works

When the application starts, FastAPI performs the following steps:

1. Inspects all route handler function signatures to extract parameter types.
2. Generates a complete OpenAPI 3.1.0 schema (accessible at `/openapi.json`).
3. Registers Pydantic models for request validation and response serialization.
4. Mounts the Swagger UI and ReDoc documentation endpoints.

Each incoming request goes through this pipeline: ASGI server receives the request, Starlette routes it to the correct handler, Pydantic validates the input data, the handler executes, and the response is serialized back through Pydantic before being sent to the client.

## Installation

Install FastAPI and an ASGI server:

```bash
pip install fastapi[standard]
```

This installs FastAPI along with Uvicorn (the recommended ASGI server), python-multipart for form data support, and httpx for the test client. The `[standard]` extra includes 6 additional packages beyond the base installation. If you prefer a minimal install, use `pip install fastapi` which installs only FastAPI, Starlette, and Pydantic.

FastAPI requires Python 3.7 or higher, though Python 3.10+ is recommended to take advantage of modern type hint syntax such as `X | None` instead of `Optional[X]`.
