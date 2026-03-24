# Dependency Injection in FastAPI

FastAPI includes a built-in dependency injection system that allows you to share logic, enforce authentication, manage database connections, and more. Dependencies are declared using `Depends()` and are resolved automatically for each request.

## Basic Dependency

A dependency is any callable (function or class) that FastAPI calls before the route handler:

```python
from fastapi import FastAPI, Depends, Query

app = FastAPI()

async def common_parameters(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    return {"skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return {"params": commons}

@app.get("/users/")
async def read_users(commons: dict = Depends(common_parameters)):
    return {"params": commons}
```

Both `/items/` and `/users/` share the same pagination logic. The `common_parameters` function is called once per request, and its return value is injected into the `commons` parameter.

## Class-Based Dependencies

Classes work as dependencies because calling a class creates an instance (i.e., `MyClass()` is callable):

```python
class PaginationParams:
    def __init__(
        self,
        skip: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(pagination: PaginationParams = Depends(PaginationParams)):
    return {"skip": pagination.skip, "limit": pagination.limit}
```

FastAPI provides a shorthand: `Depends(PaginationParams)` can also be written as `Depends()` when the type annotation already specifies the class: `pagination: PaginationParams = Depends()`.

## Sub-Dependencies

Dependencies can depend on other dependencies, forming a chain that FastAPI resolves automatically:

```python
def query_extractor(q: str | None = None):
    return q

def query_or_default(q: str = Depends(query_extractor)):
    if not q:
        return "default_query"
    return q

@app.get("/items/")
async def read_items(query: str = Depends(query_or_default)):
    return {"query": query}
```

FastAPI resolves the dependency tree from the leaves up. In this case, `query_extractor` runs first, then `query_or_default` receives its result. The maximum depth of the dependency chain is not explicitly limited, but in practice chains deeper than 10 levels indicate a design issue.

## Dependencies with Yield (Resource Management)

Use `yield` in a dependency to run setup code before and cleanup code after the route handler executes. This is ideal for managing database sessions, file handles, or locks:

```python
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def read_items(db: Session = Depends(get_db)):
    items = db.query(Item).all()
    return items
```

The code before `yield` runs before the handler, the yielded value is injected as the dependency, and the code after `yield` runs after the response is sent. The `finally` block ensures cleanup happens even if an exception occurs. FastAPI supports up to 32 yield dependencies per request by default.

## Global Dependencies

Apply dependencies to every route in the application by passing them to the `FastAPI` constructor:

```python
from fastapi import FastAPI, Depends, Header, HTTPException

async def verify_api_key(x_api_key: str = Header()):
    if x_api_key != "secret-key-123":
        raise HTTPException(status_code=403, detail="Invalid API key")

app = FastAPI(dependencies=[Depends(verify_api_key)])

@app.get("/items/")
async def read_items():
    return [{"item": "Widget"}]
```

Every route in this application requires a valid `X-Api-Key` header. You can also scope dependencies to a specific router using `APIRouter(dependencies=[...])`.

## Caching Behavior

By default, if the same dependency is used multiple times within a single request (e.g., both a route and a sub-dependency use `Depends(get_db)`), FastAPI caches the result and calls the dependency only once. To disable caching and force a fresh call each time, use `Depends(get_db, use_cache=False)`.
