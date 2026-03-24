# Error Handling in FastAPI

FastAPI provides a structured approach to error handling using HTTP exceptions, custom exception handlers, and validation error customization. Proper error handling ensures clients receive meaningful, consistent error responses.

## HTTPException

The `HTTPException` class is the primary way to return error responses from route handlers:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"widget": {"name": "Widget", "price": 35.99}}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error-Code": "ITEM_NOT_FOUND"},
        )
    return items[item_id]
```

When raised, `HTTPException` immediately terminates request processing and returns the specified status code and detail message. The `detail` parameter can be a string, list, or dictionary -- FastAPI serializes it to JSON automatically. The optional `headers` parameter adds custom HTTP headers to the error response.

The default error response format is:

```json
{
    "detail": "Item not found"
}
```

## Custom Exception Handlers

Register custom handlers for any exception type using `@app.exception_handler()`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class ItemNotFoundException(Exception):
    def __init__(self, item_id: str):
        self.item_id = item_id

app = FastAPI()

@app.exception_handler(ItemNotFoundException)
async def item_not_found_handler(request: Request, exc: ItemNotFoundException):
    return JSONResponse(
        status_code=404,
        content={
            "error": "item_not_found",
            "message": f"Item '{exc.item_id}' does not exist",
            "path": str(request.url),
        },
    )

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items_db:
        raise ItemNotFoundException(item_id)
    return items_db[item_id]
```

Custom exception handlers receive the `Request` object and the exception instance. They must return a `Response` object (typically `JSONResponse`). You can register handlers for any Python exception class, including built-in exceptions like `ValueError` or `RuntimeError`.

## Handling Validation Errors

FastAPI automatically returns a 422 Unprocessable Entity response when request validation fails. The default response includes detailed error information:

```json
{
    "detail": [
        {
            "type": "int_parsing",
            "loc": ["path", "item_id"],
            "msg": "Input should be a valid integer, unable to parse string as an integer",
            "input": "abc",
            "url": "https://errors.pydantic.dev/2/v/int_parsing"
        }
    ]
}
```

Each error object contains 5 fields: `type` (the error type identifier), `loc` (the location as a list like `["body", "price"]` or `["query", "limit"]`), `msg` (a human-readable message), `input` (the invalid value), and `url` (a link to Pydantic's error documentation).

To customize validation error responses, override the `RequestValidationError` handler:

```python
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    error_messages = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        error_messages.append(f"{field}: {error['msg']}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": error_messages,
            "error_count": len(exc.errors()),
        },
    )
```

## Overriding Default Exception Handlers

FastAPI has built-in handlers for `HTTPException` and `RequestValidationError`. You can override both:

```python
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

app = FastAPI()

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
        },
    )
```

Note: FastAPI's `HTTPException` inherits from Starlette's `HTTPException`. To override the handler for all HTTP exceptions (including those raised by Starlette internals like 404 for missing routes), register the handler for `StarletteHTTPException` rather than FastAPI's version.

## Returning the Request Body in Errors

The `RequestValidationError` object contains the original request body, which can be useful for logging or debugging:

```python
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,  # The raw request body that failed validation
        },
    )
```

The `exc.body` attribute holds the parsed request body (as a Python object) before validation was applied. This is only available for body validation errors, not for path or query parameter errors.
