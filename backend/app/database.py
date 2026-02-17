import ssl as ssl_module
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Build connect_args for asyncpg SSL support (Supabase requires SSL)
_connect_args: dict = {}
if "supabase" in settings.database_url or "ssl=require" in settings.database_url:
    # asyncpg needs an ssl.SSLContext, not just ssl=require in the URL
    _ssl_ctx = ssl_module.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = ssl_module.CERT_NONE
    _connect_args["ssl"] = _ssl_ctx

# Strip ?ssl=require from URL since asyncpg doesn't understand it as a query param
_db_url = settings.database_url.replace("?ssl=require", "").replace("&ssl=require", "")

engine = create_async_engine(
    _db_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=300,
    echo=False,
    connect_args=_connect_args,
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
