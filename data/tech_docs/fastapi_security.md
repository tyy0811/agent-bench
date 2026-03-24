# Security and Authentication in FastAPI

FastAPI provides integrated security utilities built on top of OpenAPI standards. It supports OAuth2, API keys, HTTP Basic/Bearer authentication, and OpenID Connect, with each scheme automatically reflected in the interactive documentation.

## OAuth2 with Password Flow

The most common authentication pattern uses OAuth2 "password" flow with JWT tokens:

```python
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

app = FastAPI()

SECRET_KEY = "your-secret-key-at-least-32-characters-long"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    email: str | None = None
    disabled: bool = False

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user_from_db(username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}
```

The `OAuth2PasswordBearer(tokenUrl="token")` declaration tells FastAPI that the client obtains a token by sending credentials to the `/token` endpoint. The `tokenUrl` is relative to the application root. The token is then sent in subsequent requests via the `Authorization: Bearer <token>` header.

## HTTP Bearer Authentication

For simpler token-based auth without the full OAuth2 flow:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.get("/protected/")
async def protected_route(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    token = credentials.credentials
    # validate token
    return {"token": token, "scheme": credentials.scheme}
```

`HTTPBearer` extracts the token from the `Authorization: Bearer <token>` header. If the header is missing or does not use the Bearer scheme, FastAPI returns a 403 Forbidden response automatically.

## API Key Authentication

API keys can be passed via headers, query parameters, or cookies:

```python
from fastapi.security import APIKeyHeader, APIKeyQuery

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

async def get_api_key(
    header_key: str | None = Depends(api_key_header),
    query_key: str | None = Depends(api_key_query),
):
    if header_key == "valid-api-key-12345":
        return header_key
    if query_key == "valid-api-key-12345":
        return query_key
    raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/data/", dependencies=[Depends(get_api_key)])
async def read_data():
    return {"data": "sensitive information"}
```

The `auto_error=True` parameter (the default) causes FastAPI to return an automatic 403 error when the key is missing. Setting `auto_error=False` allows the dependency to return `None` instead, letting you check multiple sources.

## OAuth2 Scopes

Scopes provide fine-grained permission control:

```python
from fastapi.security import SecurityScopes

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "items:read": "Read items",
        "items:write": "Create and update items",
        "admin": "Full administrative access",
    },
)

async def get_current_user(
    security_scopes: SecurityScopes,
    token: str = Depends(oauth2_scheme),
):
    # Decode token and verify required scopes
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    token_scopes = payload.get("scopes", [])
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(status_code=403, detail="Not enough permissions")
    return get_user_from_db(payload.get("sub"))

@app.get("/items/", dependencies=[Depends(Security(get_current_user, scopes=["items:read"]))])
async def read_items():
    return [{"item": "Widget"}]
```

Each endpoint declares the scopes it requires, and the dependency verifies the token contains all necessary permissions before allowing access.
