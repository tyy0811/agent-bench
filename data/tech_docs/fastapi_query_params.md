# Query Parameters in FastAPI

Query parameters are the key-value pairs that appear after the `?` in a URL (e.g., `/items?skip=0&limit=10`). In FastAPI, any function parameter that is not part of the path is automatically interpreted as a query parameter.

## Basic Query Parameters

```python
from fastapi import FastAPI

app = FastAPI()

# Sample data
fake_items_db = [{"item_name": f"Item {i}"} for i in range(100)]

@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

In this example, both `skip` and `limit` are query parameters with default values. A request to `/items/` uses the defaults (`skip=0`, `limit=10`), while `/items/?skip=20&limit=5` overrides both. FastAPI automatically converts the string values from the URL into their declared Python types.

## Required vs Optional Query Parameters

The distinction between required and optional query parameters depends on whether a default value is provided:

```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/search/")
async def search(
    q: str,                          # Required - no default
    category: str = "all",           # Optional - has default
    max_price: Optional[float] = None,  # Optional - default is None
):
    results = {"q": q, "category": category}
    if max_price is not None:
        results["max_price"] = max_price
    return results
```

If a client calls `/search/` without the `q` parameter, FastAPI returns a 422 Unprocessable Entity error. The `category` parameter defaults to `"all"`, and `max_price` defaults to `None`.

## Query Parameter Validation with Query()

The `Query()` function provides additional validation and metadata:

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(
    q: str = Query(
        default=None,
        min_length=3,
        max_length=50,
        pattern="^[a-zA-Z0-9 ]+$",
        title="Search query",
        description="The search string to filter items by name",
        example="laptop",
    )
):
    results = {"items": []}
    if q:
        results["q"] = q
    return results
```

The `Query()` function supports the following validation parameters for strings:

- `min_length` -- minimum character length (default: `None`, no minimum)
- `max_length` -- maximum character length (default: `None`, no maximum)
- `pattern` -- a regular expression the value must match

For numeric query parameters, `Query()` supports the same `gt`, `ge`, `lt`, and `le` constraints as `Path()`.

## Multiple Values for a Single Query Parameter

To accept a list of values for one query parameter (e.g., `/items/?tag=food&tag=drink`), declare the parameter as a `list`:

```python
from typing import List
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/items/")
async def read_items(
    tags: List[str] = Query(default=[], description="Filter by tags")
):
    return {"tags": tags}
```

A request to `/items/?tags=food&tags=drink` yields `tags=["food", "drink"]`. The default is an empty list if no tags are provided.

## Combining Path and Query Parameters

Path and query parameters work together seamlessly. FastAPI distinguishes them based on whether the parameter name appears in the path template:

```python
@app.get("/users/{user_id}/items/")
async def read_user_items(
    user_id: int,              # Path parameter (in URL path)
    skip: int = 0,             # Query parameter (not in path)
    limit: int = 10,           # Query parameter (not in path)
    include_archived: bool = False,  # Query parameter
):
    return {
        "user_id": user_id,
        "skip": skip,
        "limit": limit,
        "include_archived": include_archived,
    }
```

A request to `/users/42/items/?skip=5&limit=20&include_archived=true` passes `user_id=42` from the path and all other values from the query string. Boolean query parameters accept `true`, `false`, `1`, `0`, `yes`, `no`, `on`, and `off` (case-insensitive). FastAPI converts all these values to Python `bool`.

## Deprecating Query Parameters

You can mark a query parameter as deprecated to signal to API consumers that it will be removed in a future version:

```python
@app.get("/items/")
async def read_items(
    q: str = Query(default=None, deprecated=True)
):
    return {"q": q}
```

The parameter still functions normally, but it appears as deprecated in the generated OpenAPI documentation and Swagger UI.
