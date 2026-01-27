"""
Configuration management for Nexus backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# RAPTOR Configuration
RAPTOR_MAX_LAYERS = int(os.getenv("RAPTOR_MAX_LAYERS", "2"))
RAPTOR_CLUSTER_MIN_SIZE = int(os.getenv("RAPTOR_CLUSTER_MIN_SIZE", "5"))
RAPTOR_CHUNK_SIZE = int(os.getenv("RAPTOR_CHUNK_SIZE", "500"))
RAPTOR_CHUNK_OVERLAP = int(os.getenv("RAPTOR_CHUNK_OVERLAP", "50"))

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Gemini Models
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_GENERATION_MODEL = "gemini-2.0-flash-exp"

print(f"✓ Config loaded: ChromaDB at {CHROMA_PERSIST_DIR}")
print(f"✓ RAPTOR: {RAPTOR_MAX_LAYERS} layers, chunk size {RAPTOR_CHUNK_SIZE}")
