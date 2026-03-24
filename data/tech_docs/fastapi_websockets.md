# WebSockets in FastAPI

FastAPI supports WebSocket connections through Starlette's WebSocket implementation, enabling full-duplex, bidirectional communication between clients and servers. WebSockets are ideal for real-time features such as chat applications, live dashboards, and streaming updates.

## Basic WebSocket Endpoint

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        await ws.send_text(f"Echo: {data}")
```

The `@app.websocket()` decorator registers a WebSocket route. The handler receives a `WebSocket` object, which must be explicitly accepted by calling `await ws.accept()` before any data can be sent or received. The `accept()` method sends the HTTP 101 Switching Protocols response to the client.

## Send and Receive Methods

The `WebSocket` object provides several methods for communication:

| Method              | Description                              |
|---------------------|------------------------------------------|
| `receive_text()`    | Receive a text (string) message          |
| `receive_bytes()`   | Receive a binary message                 |
| `receive_json()`    | Receive and parse a JSON message         |
| `send_text(data)`   | Send a text message                      |
| `send_bytes(data)`  | Send binary data                         |
| `send_json(data)`   | Serialize and send a JSON message        |
| `close(code=1000)`  | Close the connection with a status code  |

The default close code is `1000` (normal closure). Other common codes are `1001` (going away), `1008` (policy violation), and `1011` (unexpected condition). The maximum WebSocket message size defaults to 16 MB in Uvicorn, configurable via the `--ws-max-size` flag.

## Handling Disconnects

Clients can disconnect at any time. Handle this with `WebSocketDisconnect`:

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def chat_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            await manager.broadcast(f"User says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(ws)
        await manager.broadcast("A user has left the chat")
```

The `WebSocketDisconnect` exception is raised when `receive_text()`, `receive_bytes()`, or `receive_json()` detects that the client has closed the connection. The exception has a `code` attribute containing the close code sent by the client.

## WebSocket with Path Parameters and Dependencies

WebSocket endpoints support path parameters, query parameters, and dependency injection:

```python
from fastapi import FastAPI, WebSocket, Depends, Query, Path, Cookie, Header

app = FastAPI()

async def get_token(
    websocket: WebSocket,
    token: str | None = Query(default=None),
    x_token: str | None = Header(default=None),
):
    if token is None and x_token is None:
        await websocket.close(code=1008)
        return None
    return token or x_token

@app.websocket("/ws/{room_id}")
async def room_websocket(
    ws: WebSocket,
    room_id: int = Path(ge=1, le=1000),
    token: str | None = Depends(get_token),
):
    if token is None:
        return
    await ws.accept()
    await ws.send_text(f"Connected to room {room_id}")
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"[Room {room_id}] {data}")
    except WebSocketDisconnect:
        pass
```

Dependencies for WebSocket endpoints work the same as for HTTP endpoints, including `Depends()`, `Path()`, `Query()`, `Header()`, and `Cookie()`. However, WebSocket endpoints do not support `Body()` parameters since WebSocket communication uses its own message protocol rather than HTTP request bodies.

## WebSocket with JSON Messages

For structured communication, use JSON messages with Pydantic validation:

```python
from pydantic import BaseModel, ValidationError

class ChatMessage(BaseModel):
    username: str
    content: str
    channel: str = "general"

@app.websocket("/ws/json")
async def json_websocket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw_data = await ws.receive_json()
            try:
                message = ChatMessage(**raw_data)
                await ws.send_json({
                    "status": "ok",
                    "echo": message.model_dump(),
                })
            except ValidationError as e:
                await ws.send_json({
                    "status": "error",
                    "errors": e.errors(),
                })
    except WebSocketDisconnect:
        pass
```

The `receive_json()` method parses the incoming text message as JSON. If the message is not valid JSON, it raises a `json.JSONDecodeError`. Pydantic validation is applied manually since FastAPI does not automatically validate WebSocket message payloads the way it validates HTTP request bodies.
