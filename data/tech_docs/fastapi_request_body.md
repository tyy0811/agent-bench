# Request Body in FastAPI

A request body is data sent by the client to your API, typically as JSON in POST, PUT, or PATCH requests. FastAPI uses Pydantic models to declare, validate, and serialize request bodies with full type safety.

## Defining a Request Body with Pydantic

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float = 0.0

@app.post("/items/")
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax > 0:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict
```

When a client sends a POST request with a JSON body like `{"name": "Widget", "price": 35.99, "tax": 3.60}`, FastAPI automatically parses the JSON, validates it against the `Item` model, and passes the validated object to the handler. If `description` is omitted, it defaults to `None`. If `tax` is omitted, it defaults to `0.0`. If `name` or `price` is missing, a 422 Unprocessable Entity response is returned.

## Field Validation

The `Field()` function from Pydantic lets you add constraints and metadata to individual model fields:

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(
        min_length=1,
        max_length=100,
        description="The name of the item",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="An optional text description",
    )
    price: float = Field(
        gt=0,
        le=1_000_000,
        description="Price must be greater than 0 and at most 1,000,000",
    )
    quantity: int = Field(
        default=1,
        ge=1,
        le=9999,
        description="Quantity between 1 and 9999",
    )
```

Pydantic validates all constraints at request time. The `gt`, `ge`, `lt`, `le` parameters mirror the same semantics as FastAPI's `Path()` and `Query()`. The `min_length` and `max_length` parameters work on string fields.

## Nested Models

Pydantic models can contain other models, lists, and complex nested structures:

```python
from pydantic import BaseModel, HttpUrl

class Image(BaseModel):
    url: HttpUrl
    name: str
    width: int = Field(ge=1, le=10000)
    height: int = Field(ge=1, le=10000)

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tags: list[str] = []
    images: list[Image] = []

class Offer(BaseModel):
    name: str
    description: str | None = None
    items: list[Item]
    discount_percent: float = Field(ge=0, le=100)
```

FastAPI validates the entire nested structure recursively. If any nested field fails validation, the error response includes the exact path to the invalid field (e.g., `body -> items -> 0 -> images -> 1 -> url`).

## Combining Body, Path, and Query Parameters

You can accept all three parameter types in a single endpoint:

```python
from fastapi import FastAPI, Path, Query
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.put("/items/{item_id}")
async def update_item(
    item_id: int = Path(ge=1, le=10000),
    q: str | None = Query(default=None, max_length=50),
    item: Item = ...,
):
    result = {"item_id": item_id, **item.model_dump()}
    if q:
        result["q"] = q
    return result
```

FastAPI determines the source of each parameter by these rules: if the parameter name appears in the path string, it is a path parameter; if the type is a Pydantic model (or annotated with `Body()`), it comes from the request body; otherwise, it is a query parameter.

## Multiple Body Parameters

When you need multiple distinct objects in the request body, declare multiple Pydantic model parameters:

```python
from fastapi import Body

class Item(BaseModel):
    name: str
    price: float

class User(BaseModel):
    username: str
    email: str

@app.put("/items/{item_id}")
async def update_item(
    item_id: int,
    item: Item,
    user: User,
    importance: int = Body(gt=0, le=5),
):
    return {"item_id": item_id, "item": item, "user": user, "importance": importance}
```

The expected JSON body becomes `{"item": {...}, "user": {...}, "importance": 3}`. Each model is keyed by its parameter name. The `Body()` function embeds a singular value inside the body alongside the models, rather than treating it as a query parameter. The maximum request body size is controlled by the ASGI server; Uvicorn defaults to approximately 1 MB.
