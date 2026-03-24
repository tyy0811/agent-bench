# Testing FastAPI Applications

FastAPI applications are tested using the `TestClient` class, which provides a synchronous interface for sending requests to your application without running an actual server. For async testing, use `httpx.AsyncClient`.

## Basic Testing with TestClient

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    result = {"item_id": item_id}
    if q:
        result["q"] = q
    return result

client = TestClient(app)

def test_read_item():
    response = client.get("/items/42?q=test")
    assert response.status_code == 200
    assert response.json() == {"item_id": 42, "q": "test"}

def test_read_item_not_found():
    response = client.get("/items/abc")
    assert response.status_code == 422  # Validation error
```

The `TestClient` is built on top of `httpx` (which replaced `requests` as of Starlette 0.20.0). It supports all HTTP methods: `client.get()`, `client.post()`, `client.put()`, `client.delete()`, `client.patch()`, `client.options()`, and `client.head()`.

## Pytest Fixtures

Use fixtures to share the `TestClient` and set up test data:

```python
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from myapp.main import app
from myapp.database import Base, engine

@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token-12345"}

def test_create_item(client, auth_headers):
    response = client.post(
        "/items/",
        json={"name": "Widget", "price": 35.99},
        headers=auth_headers,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert "id" in data
```

Using `scope="module"` means the fixture is created once per test module rather than once per test function, improving performance when database setup is expensive. The `with` statement ensures proper cleanup of the test client's underlying transport.

## Overriding Dependencies in Tests

Override dependencies to inject mock services or test databases:

```python
from fastapi import FastAPI, Depends

app = FastAPI()

async def get_db():
    db = ProductionDatabase()
    try:
        yield db
    finally:
        db.close()

@app.get("/items/")
async def read_items(db=Depends(get_db)):
    return db.query_all_items()

# In your test file:
def get_test_db():
    db = TestDatabase()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = get_test_db

client = TestClient(app)

def test_read_items():
    response = client.get("/items/")
    assert response.status_code == 200

# Clean up overrides after tests
app.dependency_overrides.clear()
```

The `app.dependency_overrides` dictionary maps original dependencies to their replacements. This works for any dependency in the chain, including sub-dependencies. Always call `app.dependency_overrides.clear()` after tests to prevent overrides from leaking between test modules.

## Async Testing with httpx

For testing async-specific behavior (e.g., async database calls, WebSocket-related setup), use `httpx.AsyncClient` with `pytest-asyncio`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from myapp.main import app

@pytest.mark.anyio
async def test_read_items_async():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/items/")
        assert response.status_code == 200

@pytest.mark.anyio
async def test_create_item_async():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/items/",
            json={"name": "Widget", "price": 35.99},
        )
        assert response.status_code == 201
```

The `ASGITransport` connects `httpx` directly to the ASGI application without network overhead. The `base_url` parameter is required but can be any valid URL since no real network requests are made. Install the async test dependencies with `pip install httpx pytest-asyncio` (or use `anyio` with the `@pytest.mark.anyio` marker).

## Testing WebSockets

```python
def test_websocket():
    client = TestClient(app)
    with client.websocket_connect("/ws") as websocket:
        websocket.send_text("hello")
        data = websocket.receive_text()
        assert data == "Message received: hello"
```

The `websocket_connect` context manager establishes a WebSocket connection. It supports `send_text()`, `send_json()`, `send_bytes()`, `receive_text()`, `receive_json()`, and `receive_bytes()` methods.
