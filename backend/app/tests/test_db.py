"""Quick script to verify Supabase connection and table access."""

import asyncio
import ssl as ssl_module
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings


async def test_connection():
    db_url = settings.database_url.replace("?ssl=require", "").replace("&ssl=require", "")
    print(f"Connecting to: {db_url[:40]}...")

    # SSL config for Supabase
    connect_args: dict = {}
    if "supabase" in db_url or "ssl=require" in settings.database_url:
        ctx = ssl_module.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl_module.CERT_NONE
        connect_args["ssl"] = ctx

    engine = create_async_engine(db_url, pool_pre_ping=True, connect_args=connect_args)

    try:
        async with engine.connect() as conn:
            # 1. Basic connectivity
            result = await conn.execute(text("SELECT 1"))
            print(f"✓ Connection successful: {result.scalar()}")

            # 2. Check pgvector extension
            result = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            row = result.first()
            print(f"✓ pgvector extension: {'installed' if row else '✗ NOT installed'}")

            # 3. List our tables
            result = await conn.execute(
                text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    ORDER BY table_name
                """)
            )
            tables = [row[0] for row in result.fetchall()]
            print(f"✓ Tables found ({len(tables)}):")
            for t in tables:
                print(f"    - {t}")

            # 4. Row counts for key tables
            for table in [
                "evidence", "evidence_chunks", "problem_mentions",
                "problem_clusters", "feature_proposals", "jobs",
                "llm_call_log", "tasks", "priority_scores", "proposal_versions",
            ]:
                if table in tables:
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"    {table}: {count} rows")

    except Exception as e:
        print(f"✗ Connection failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await engine.dispose()

    print("\n✓ All checks passed!")


if __name__ == "__main__":
    asyncio.run(test_connection())
