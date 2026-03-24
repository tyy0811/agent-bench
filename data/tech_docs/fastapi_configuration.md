# Configuration and Settings in FastAPI

FastAPI leverages Pydantic's `BaseSettings` class to manage application configuration through environment variables, `.env` files, and secrets. This approach provides type-safe configuration with validation, default values, and automatic environment variable reading.

## Pydantic Settings

Install the settings extension:

```bash
pip install pydantic-settings
```

Define your settings as a Pydantic model:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="",
    )

    app_name: str = "My FastAPI App"
    admin_email: str = "admin@example.com"
    debug: bool = False
    database_url: str = "sqlite:///./app.db"
    redis_url: str = "redis://localhost:6379/0"
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]
    max_connections: int = 100
    api_v1_prefix: str = "/api/v1"
    access_token_expire_minutes: int = 30
    secret_key: str = "change-me-in-production"
```

Pydantic Settings reads values from these sources in the following priority order (highest priority first):

1. Constructor arguments passed directly to `Settings()`
2. Environment variables
3. Variables from the `.env` file
4. Default values defined in the model

Setting `case_sensitive=False` (the default) means the environment variable `DATABASE_URL`, `database_url`, and `Database_Url` all map to the `database_url` field.

## Environment Variables and .env Files

Create a `.env` file in the project root:

```
APP_NAME=Production API
DEBUG=false
DATABASE_URL=postgresql://user:pass@db-host:5432/mydb
REDIS_URL=redis://redis-host:6379/0
MAX_CONNECTIONS=250
SECRET_KEY=a7f3b9c1d4e8f2a6b0c5d9e3f7a1b4c8
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

The `.env` file is parsed using the `python-dotenv` library (installed automatically with `pydantic-settings`). Multiple `.env` files can be specified as a tuple:

```python
model_config = SettingsConfigDict(
    env_file=(".env", ".env.local"),
)
```

When multiple files are specified, later files take precedence over earlier ones. So `.env.local` overrides values from `.env`.

## Settings as a Dependency

Use dependency injection to provide settings to route handlers:

```python
from functools import lru_cache
from fastapi import FastAPI, Depends

app = FastAPI()

@lru_cache
def get_settings():
    return Settings()

@app.get("/info")
async def info(settings: Settings = Depends(get_settings)):
    return {
        "app_name": settings.app_name,
        "admin_email": settings.admin_email,
        "debug": settings.debug,
    }
```

The `@lru_cache` decorator ensures the `Settings` object is created only once and reused for all subsequent requests. Without caching, Pydantic would read and parse the `.env` file on every request, adding approximately 1-3 milliseconds of overhead per call. The cache has no size limit by default (`maxsize=128` for `lru_cache`), but since `get_settings()` takes no arguments, it effectively stores just one instance.

## Nested Settings with Prefixes

Organize related settings into nested models using `env_prefix`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "mydb"
    user: str = "postgres"
    password: str = ""
    pool_min_size: int = 5
    pool_max_size: int = 20
    echo: bool = False

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    app_name: str = "My App"
    debug: bool = False
    db: DatabaseSettings = DatabaseSettings()
```

With `env_prefix="DB_"`, the environment variable `DB_HOST` maps to `DatabaseSettings.host`, `DB_PORT` maps to `port`, and so on. The default database pool sizes are 5 minimum and 20 maximum connections.

## Secrets Management

For sensitive values, Pydantic Settings supports reading from secret files (commonly used with Docker Secrets and Kubernetes Secrets):

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        secrets_dir="/run/secrets",
    )

    database_password: str
    api_key: str
    jwt_secret: str
```

When `secrets_dir` is set, Pydantic looks for files named after each field (e.g., `/run/secrets/database_password`). The file contents become the field value. Secret files take precedence over `.env` values but are overridden by environment variables.

The priority order with secrets becomes: constructor arguments > environment variables > secret files > `.env` file > default values.
