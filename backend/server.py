"""
Nexus API Server

FastAPI server exposing the RAG system and Agentic AI endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.registry import AgentRegistry
from agents.routes import create_agents_router

app = FastAPI(
    title="Nexus API",
    description="AI-powered document research workspace with agentic AI",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent registry (singleton)
agent_registry = AgentRegistry()

# Mount agent routes
app.include_router(create_agents_router(agent_registry))


@app.get("/")
def root():
    return {
        "name": "Nexus API",
        "version": "0.1.0",
        "features": ["rag", "agents", "orchestrator"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
