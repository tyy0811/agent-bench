# Response Model in FastAPI

The `response_model` parameter on route decorators lets you declare the shape of the data your endpoint returns. FastAPI uses it to validate, serialize, and document the response -- filtering out any fields not defined in the model and generating accurate OpenAPI schemas.

## Basic Response Model

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserIn(BaseModel):
    username: str
    email: str
    password: str

class UserOut(BaseModel):
    username: str
    email: str

@app.post("/users/", response_model=UserOut, status_code=201)
async def create_user(user: UserIn):
    # In a real app, hash the password and save to DB
    return user  # password is automatically filtered out
```

Even though the handler returns the full `UserIn` object (which includes `password`), the `response_model=UserOut` declaration ensures that only `username` and `email` appear in the response. This is a critical security pattern -- it prevents accidental leakage of sensitive fields like passwords, tokens, or internal IDs.

## Status Codes

FastAPI provides the `status_code` parameter to set the HTTP response status code. Common codes include:

| Code | Constant                           | Usage                |
|------|------------------------------------|----------------------|
| 200  | `status.HTTP_200_OK`               | Successful GET       |
| 201  | `status.HTTP_201_CREATED`          | Successful creation  |
| 204  | `status.HTTP_204_NO_CONTENT`       | Successful deletion  |
| 400  | `status.HTTP_400_BAD_REQUEST`      | Client error         |
| 404  | `status.HTTP_404_NOT_FOUND`        | Resource not found   |
| 422  | `status.HTTP_422_UNPROCESSABLE_ENTITY` | Validation error |

```python
from fastapi import status

@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: int):
    # delete logic
    return None
```

The default `status_code` for all route decorators is `200`.

## Filtering Fields with response_model_include and response_model_exclude

You can dynamically control which fields appear in the response without creating a separate model:

```python
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 0.0
    internal_code: str = "N/A"

@app.get(
    "/items/{item_id}",
    response_model=Item,
    response_model_exclude={"internal_code"},
)
async def read_item(item_id: int):
    return {
        "name": "Widget",
        "description": "A useful widget",
        "price": 35.99,
        "tax": 3.60,
        "internal_code": "WDG-001",
    }
```

The `response_model_exclude` parameter accepts a `set` of field names to strip from the output. Similarly, `response_model_include` accepts a `set` of field names to keep -- all others are excluded. If both are provided, `response_model_include` is applied first, then `response_model_exclude` removes fields from that subset.

## Excluding Unset and Default Values

Two additional parameters control whether default or unset values appear in the response:

```python
@app.get(
    "/items/{item_id}",
    response_model=Item,
    response_model_exclude_unset=True,
)
async def read_item(item_id: int):
    return Item(name="Widget", price=35.99)
    # Response: {"name": "Widget", "price": 35.99}
    # Fields with defaults (description, tax) are omitted
```

- `response_model_exclude_unset=True` -- omits fields the user did not explicitly set (default: `False`)
- `response_model_exclude_defaults=True` -- omits fields whose value matches the default (default: `False`)
- `response_model_exclude_none=True` -- omits fields with `None` values (default: `False`)

## Multiple Response Models

Use `Union` types or the `responses` parameter to document endpoints that may return different shapes:

```python
from typing import Union

class ItemPublic(BaseModel):
    name: str
    price: float

class ItemAdmin(BaseModel):
    name: str
    price: float
    internal_code: str
    profit_margin: float

@app.get("/items/{item_id}", response_model=Union[ItemAdmin, ItemPublic])
async def read_item(item_id: int, is_admin: bool = False):
    item_data = get_item(item_id)
    if is_admin:
        return ItemAdmin(**item_data)
    return ItemPublic(**item_data)
```

When using `Union`, Pydantic validates the response against each model in order and uses the first match. Place the more specific model first (the one with more fields) to avoid premature matching.
