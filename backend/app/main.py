from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import evidence, jobs

app = FastAPI(title="Nexus Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evidence.router, prefix="/api/v1", tags=["evidence"])
app.include_router(jobs.router, prefix="/api/v1", tags=["jobs"])


@app.get("/api/v1/health")
async def health_check() -> dict:
    return {"status": "ok"}
