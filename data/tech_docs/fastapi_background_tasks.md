# Background Tasks in FastAPI

Background tasks allow you to schedule work to run after the response has been sent to the client. This is useful for operations that do not need to complete before the user receives a response, such as sending emails, writing audit logs, or triggering data processing pipelines.

## Basic Background Task

```python
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

def write_log(message: str):
    with open("log.txt", "a") as f:
        f.write(f"{message}\n")

@app.post("/items/", status_code=201)
async def create_item(name: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, f"Item created: {name}")
    return {"name": name, "status": "created"}
```

Declare `BackgroundTasks` as a parameter in your route handler, and FastAPI injects it automatically. Call `add_task()` with the function to run and any positional or keyword arguments. The task runs after the response is sent, in the same process. The `add_task()` method accepts both synchronous and asynchronous functions -- sync functions are run in a threadpool, while async functions are awaited on the event loop.

## Multiple Background Tasks

You can add multiple tasks, and they execute sequentially in the order they were added:

```python
def send_email(to: str, subject: str, body: str):
    # Simulate sending email (takes ~2 seconds)
    import time
    time.sleep(2)
    print(f"Email sent to {to}: {subject}")

def update_analytics(event: str, item_id: int):
    # Record analytics event
    print(f"Analytics: {event} for item {item_id}")

@app.post("/items/{item_id}/purchase")
async def purchase_item(item_id: int, background_tasks: BackgroundTasks):
    # Process purchase immediately
    result = process_purchase(item_id)

    # Queue background work
    background_tasks.add_task(
        send_email,
        to="buyer@example.com",
        subject="Purchase Confirmation",
        body=f"You purchased item {item_id}",
    )
    background_tasks.add_task(update_analytics, "purchase", item_id)

    return {"item_id": item_id, "status": "purchased"}
```

In this example, the client receives the response immediately after purchase processing. The email and analytics tasks run sequentially in the background. If the first task takes 2 seconds, the second task starts only after the first completes.

## Background Tasks in Dependencies

Dependencies can also add background tasks, which is useful for cross-cutting concerns like logging:

```python
from fastapi import Depends

def log_request(background_tasks: BackgroundTasks):
    def _log(method: str, path: str):
        with open("access.log", "a") as f:
            f.write(f"{method} {path}\n")
    return background_tasks, _log

async def audit_dependency(
    background_tasks: BackgroundTasks,
    request_method: str = "GET",
):
    def audit_log(action: str):
        with open("audit.log", "a") as f:
            f.write(f"[{request_method}] {action}\n")
    background_tasks.add_task(audit_log, "endpoint_accessed")

@app.get("/items/", dependencies=[Depends(audit_dependency)])
async def read_items(background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, "Items listed")
    return [{"item": "Widget"}]
```

When both the dependency and the route handler add tasks to `BackgroundTasks`, all tasks share the same task queue. Dependency tasks are added first (in the order dependencies are resolved), followed by tasks added in the route handler.

## Use Cases and Limitations

Common use cases for background tasks:

- **Email notifications**: Send confirmation or alert emails after an action (typical send time: 1-5 seconds).
- **Log writing**: Write detailed audit logs without adding latency to the response.
- **Cache invalidation**: Clear or update caches after data mutations.
- **Webhook delivery**: POST event payloads to external services with retry logic.
- **File cleanup**: Remove temporary uploaded files after processing.

Important limitations to consider:

1. Background tasks run in the same process as the web server. If a task crashes, it does not affect the already-sent response, but unhandled exceptions are logged to stderr.
2. If the server shuts down, pending background tasks are lost -- they are not persisted to a queue. For critical tasks, use a dedicated task queue like Celery (which supports up to 10,000+ tasks per second with Redis as a broker) or ARQ.
3. Background tasks share the event loop (for async tasks) or threadpool (for sync tasks, default pool size of 40 threads). A CPU-intensive background task can degrade request handling performance.
4. There is no built-in retry mechanism. If a background task fails, it fails silently from the client's perspective. Implement retry logic within the task function if needed.
