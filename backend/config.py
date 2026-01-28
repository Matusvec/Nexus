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

# Document Processing (Production Settings)
# Skip tiny images (icons, artifacts) - most PDFs have junk images
MIN_IMAGE_SIZE_BYTES = int(os.getenv("MIN_IMAGE_SIZE_BYTES", "5000"))  # 5KB minimum
PROCESS_IMAGES = os.getenv("PROCESS_IMAGES", "false").lower() == "true"  # Disabled by default

# Entity Extraction Mode: "llm" (slow but accurate) or "fast" (spaCy/rules, 1000x faster)
ENTITY_EXTRACTION_MODE = os.getenv("ENTITY_EXTRACTION_MODE", "fast")

# Gemini Models
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_GENERATION_MODEL = "gemini-2.5-flash"

print(f"[OK] Config loaded: ChromaDB at {CHROMA_PERSIST_DIR}")
print(f"[OK] RAPTOR: {RAPTOR_MAX_LAYERS} layers, chunk size {RAPTOR_CHUNK_SIZE}")
