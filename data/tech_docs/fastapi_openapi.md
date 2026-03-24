# OpenAPI and Documentation in FastAPI

FastAPI automatically generates an OpenAPI 3.1.0 schema from your code, providing interactive documentation interfaces with zero configuration. This schema drives Swagger UI and ReDoc, and can be consumed by code generators, API gateways, and testing tools.

## Automatic Documentation Endpoints

Every FastAPI application exposes three documentation-related endpoints by default:

| Endpoint         | Description                                      |
|------------------|--------------------------------------------------|
| `/docs`          | Swagger UI -- interactive API explorer           |
| `/redoc`         | ReDoc -- alternative documentation viewer        |
| `/openapi.json`  | Raw OpenAPI schema in JSON format                |

```python
from fastapi import FastAPI

app = FastAPI(
    title="My API",
    description="A comprehensive API for managing items and users.",
    version="2.1.0",
    terms_of_service="https://example.com/terms",
    contact={
        "name": "API Support",
        "url": "https://example.com/support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)
```

To disable any documentation endpoint, set its URL to `None`:

```python
app = FastAPI(
    docs_url=None,      # Disables Swagger UI
    redoc_url=None,     # Disables ReDoc
    openapi_url=None,   # Disables OpenAPI schema (also disables both UIs)
)
```

Disabling `openapi_url` effectively disables all automatic documentation since both Swagger UI and ReDoc depend on the OpenAPI schema.

## Tags and Grouping

Organize endpoints into logical groups using tags:

```python
from fastapi import FastAPI

tags_metadata = [
    {
        "name": "users",
        "description": "Operations with users. The **login** logic is also here.",
    },
    {
        "name": "items",
        "description": "Manage items. Each item has a unique integer ID.",
        "externalDocs": {
            "description": "Items external docs",
            "url": "https://example.com/items-docs",
        },
    },
]

app = FastAPI(openapi_tags=tags_metadata)

@app.get("/users/", tags=["users"])
async def read_users():
    return [{"username": "alice"}]

@app.get("/items/", tags=["items"])
async def read_items():
    return [{"name": "Widget"}]

@app.post("/items/", tags=["items"])
async def create_item(name: str):
    return {"name": name}
```

Tags appear as collapsible sections in Swagger UI. The order of tags in `openapi_tags` determines their display order. An endpoint can have multiple tags, causing it to appear in each corresponding section.

## Enriching Endpoint Documentation

Add descriptions, summaries, and response documentation to individual endpoints:

```python
from fastapi import FastAPI, Path, Query
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    """An item in the inventory system."""
    name: str
    price: float
    description: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Premium Widget",
                    "price": 35.99,
                    "description": "A high-quality widget",
                }
            ]
        }
    }

@app.get(
    "/items/{item_id}",
    summary="Get a single item",
    description="Retrieve an item by its unique integer ID. Returns 404 if the item does not exist.",
    response_description="The requested item with all fields populated",
    deprecated=False,
    operation_id="getItemById",
)
async def read_item(
    item_id: int = Path(
        title="Item ID",
        description="The unique identifier for the item",
        ge=1,
        example=42,
    ),
):
    return {"item_id": item_id, "name": "Widget", "price": 35.99}
```

If no `summary` is provided, FastAPI uses the function name converted to title case (e.g., `read_item` becomes "Read Item"). If no `description` is provided, FastAPI uses the function's docstring.

The `operation_id` sets a unique identifier for the endpoint in the OpenAPI schema. By default, FastAPI generates operation IDs by combining the HTTP method and function name (e.g., `read_item_items__item_id__get`). Custom operation IDs are useful when generating client SDKs.

## Customizing the OpenAPI Schema

Override or extend the generated schema programmatically:

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Custom API",
        version="3.0.0",
        summary="An API with a custom OpenAPI schema",
        description="This schema includes additional vendor extensions.",
        routes=app.routes,
    )

    # Add custom vendor extension
    openapi_schema["x-api-audience"] = "public"

    # Modify schema components
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png",
        "altText": "API Logo",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

The `get_openapi()` function generates the base schema from the application's routes. By assigning the result to `app.openapi_schema`, you cache it so it is only generated once. The cached schema is served at `/openapi.json` for all subsequent requests.

## Multiple Examples

Provide multiple request/response examples for a single endpoint:

```python
from fastapi import Body

@app.post("/items/")
async def create_item(
    item: Item = Body(
        openapi_examples={
            "minimal": {
                "summary": "Minimal item",
                "description": "Only required fields",
                "value": {"name": "Widget", "price": 9.99},
            },
            "complete": {
                "summary": "Complete item",
                "description": "All fields populated",
                "value": {
                    "name": "Premium Widget",
                    "price": 35.99,
                    "description": "A high-quality widget",
                },
            },
        },
    ),
):
    return item
```

These examples appear in Swagger UI as a dropdown menu, allowing API consumers to quickly test different request scenarios. Each example requires a `summary` and `value`; the `description` is optional.
