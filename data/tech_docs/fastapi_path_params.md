# Path Parameters in FastAPI

Path parameters allow you to capture variable segments of a URL path and pass them directly to your route handler function. They are declared using curly braces `{}` in the route path string and must have a corresponding parameter in the function signature.

## Basic Path Parameters

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}
```

When a client sends a request to `/users/42`, FastAPI will automatically parse `"42"` from the URL, validate that it can be converted to an `int`, and pass `user_id=42` to the handler. If the client sends `/users/abc`, FastAPI returns a 422 Unprocessable Entity response with a detailed validation error.

## Type Annotations and Validation

Path parameters support all standard Python types for automatic conversion:

- `int` -- integer values (e.g., `/items/5`)
- `float` -- floating-point values (e.g., `/prices/9.99`)
- `str` -- string values (this is the default if no type is specified)
- `bool` -- boolean values, accepts `true`, `false`, `1`, `0`, `yes`, `no`
- `uuid.UUID` -- UUID strings (e.g., `/records/550e8400-e29b-41d4-a716-446655440000`)

## Path Parameter Validation with Path()

Use the `Path()` function from FastAPI to add validation constraints:

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(
    item_id: int = Path(
        title="The ID of the item",
        description="A unique integer identifier",
        ge=1,
        le=10000,
    )
):
    return {"item_id": item_id}
```

The `Path()` function supports these numeric validation parameters:

| Parameter | Meaning               | Example |
|-----------|-----------------------|---------|
| `gt`      | greater than          | `gt=0`  |
| `ge`      | greater than or equal | `ge=1`  |
| `lt`      | less than             | `lt=100`|
| `le`      | less than or equal    | `le=99` |

For string path parameters, you can use `min_length` and `max_length` constraints. The default `min_length` is `None` (no minimum), and the maximum allowed `max_length` for path parameters in practice is 255 characters due to URL length limitations in most web servers.

## Route Order Matters

FastAPI evaluates routes in the order they are defined. This is critical when you have routes that could match the same URL pattern:

```python
@app.get("/users/me")
async def read_current_user():
    return {"user": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}
```

The `/users/me` route **must** be declared before `/users/{user_id}`. If the order is reversed, a request to `/users/me` would match the parameterized route first, and `user_id` would receive the string `"me"` as its value instead of triggering the dedicated handler.

## Enum Path Parameters

You can restrict path parameters to a fixed set of values using Python's `Enum`:

```python
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    return {"model_name": model_name, "message": "Other model selected"}
```

If the client sends a value not in the enum, FastAPI returns a 422 response listing all permitted values.

## File Path Parameters

To capture an entire file path (including slashes), use the `:path` converter:

```python
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    return {"file_path": file_path}
```

A request to `/files/home/user/data.csv` will set `file_path` to `"home/user/data.csv"`.
