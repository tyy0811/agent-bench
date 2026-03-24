# Pagination in FastAPI

Pagination is essential for any API that returns collections of resources. Without pagination, endpoints serving large datasets would consume excessive memory, bandwidth, and time. FastAPI supports multiple pagination strategies, each suited to different use cases.

## Offset/Limit Pagination (Skip/Limit Pattern)

The most common approach uses `skip` and `limit` query parameters:

```python
from fastapi import FastAPI, Query, Depends
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float

# Simulated database with 10,000 items
all_items = [Item(id=i, name=f"Item {i}", price=round(i * 1.5, 2)) for i in range(1, 10001)]

class PaginationParams:
    def __init__(
        self,
        skip: int = Query(default=0, ge=0, description="Number of items to skip"),
        limit: int = Query(default=20, ge=1, le=100, description="Number of items to return"),
    ):
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def list_items(pagination: PaginationParams = Depends()):
    items = all_items[pagination.skip : pagination.skip + pagination.limit]
    return {
        "items": items,
        "total": len(all_items),
        "skip": pagination.skip,
        "limit": pagination.limit,
    }
```

This implementation uses a default page size of 20 items, a minimum of 1 item per page, and a maximum of 100 items per page. For a dataset of 10,000 items with the default page size of 20, there are 500 total pages. Requesting page 3 would use `skip=40&limit=20` to retrieve items 41 through 60.

The offset/limit pattern is simple to implement but has performance drawbacks for large offsets. A query with `skip=9000` on a SQL database must scan and discard 9,000 rows before returning the requested 20, resulting in O(n) performance where n is the offset value.

## Cursor-Based Pagination

Cursor-based pagination uses an opaque token (cursor) pointing to the last item in the previous page. This avoids the performance degradation of large offsets:

```python
import base64
from fastapi import FastAPI, Query

app = FastAPI()

def encode_cursor(item_id: int) -> str:
    return base64.urlsafe_b64encode(f"id:{item_id}".encode()).decode()

def decode_cursor(cursor: str) -> int:
    decoded = base64.urlsafe_b64decode(cursor.encode()).decode()
    return int(decoded.split(":")[1])

@app.get("/items/")
async def list_items(
    cursor: str | None = Query(default=None, description="Pagination cursor"),
    limit: int = Query(default=20, ge=1, le=100),
):
    if cursor:
        last_id = decode_cursor(cursor)
        # In a real DB: SELECT * FROM items WHERE id > last_id ORDER BY id LIMIT limit
        items = [item for item in all_items if item.id > last_id][:limit]
    else:
        items = all_items[:limit]

    next_cursor = None
    if len(items) == limit:
        next_cursor = encode_cursor(items[-1].id)

    return {
        "items": items,
        "next_cursor": next_cursor,
        "limit": limit,
        "has_more": len(items) == limit,
    }
```

Cursor-based pagination maintains consistent O(1) performance regardless of how deep into the dataset the client has paginated. It is the recommended approach for datasets exceeding 100,000 records or for real-time feeds where items may be inserted or deleted between page requests.

## Pagination with Total Count and Link Headers

Include total count metadata and RFC 5988 Link headers for discoverability:

```python
from fastapi import FastAPI, Query, Response
from math import ceil

app = FastAPI()

@app.get("/items/")
async def list_items(
    response: Response,
    page: int = Query(default=1, ge=1, description="Page number"),
    per_page: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    total = len(all_items)
    total_pages = ceil(total / per_page)
    skip = (page - 1) * per_page
    items = all_items[skip : skip + per_page]

    # Build Link headers
    base_url = "/items/"
    links = []
    if page > 1:
        links.append(f'<{base_url}?page=1&per_page={per_page}>; rel="first"')
        links.append(f'<{base_url}?page={page - 1}&per_page={per_page}>; rel="prev"')
    if page < total_pages:
        links.append(f'<{base_url}?page={page + 1}&per_page={per_page}>; rel="next"')
        links.append(f'<{base_url}?page={total_pages}&per_page={per_page}>; rel="last"')

    response.headers["Link"] = ", ".join(links)
    response.headers["X-Total-Count"] = str(total)
    response.headers["X-Total-Pages"] = str(total_pages)

    return {
        "items": items,
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    }
```

With 10,000 items and a default page size of 20, the `X-Total-Pages` header returns 500. At 50 items per page, there are 200 total pages. The Link header follows the RFC 5988 standard used by the GitHub API and other major REST APIs.

## Pagination Response Model

Standardize pagination responses across endpoints with a generic response model:

```python
from typing import Generic, TypeVar, List
from pydantic import BaseModel

T = TypeVar("T")

class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    per_page: int
    total_pages: int

@app.get("/items/", response_model=PaginatedResponse[Item])
async def list_items(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
):
    total = len(all_items)
    skip = (page - 1) * per_page
    return PaginatedResponse(
        items=all_items[skip : skip + per_page],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=ceil(total / per_page),
    )
```

This generic model ensures every paginated endpoint returns a consistent structure. The `total_pages` field is always calculated as `ceil(total / per_page)`. For 10,000 items at 20 per page, that is `ceil(10000 / 20) = 500` pages. For 10,000 items at 30 per page, that is `ceil(10000 / 30) = 334` pages (with the last page containing only 10 items).
