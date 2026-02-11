"""
conftest.py – shared fixtures for the T-Retriever test suite.

Patches external dependencies (Gemini API, ChromaDB) so tests run
entirely in-process without network calls or persistent state.
"""
import sys
import types
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Ensure `backend/` is on the import path
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------------
# Stub out config *before* any backend module is imported
# ---------------------------------------------------------------------------
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", "/tmp/nexus_test_chroma")

# Create a mock config module to avoid the API-key ValueError
_config = types.ModuleType("config")
_config.GEMINI_API_KEY = "test-key-not-real"
_config.CHROMA_PERSIST_DIR = "/tmp/nexus_test_chroma"
_config.COLLECTION_NAME = "test_chunks"
_config.HOST = "0.0.0.0"
_config.PORT = 8000
_config.MIN_IMAGE_SIZE_BYTES = 5000
_config.PROCESS_IMAGES = False
_config.ENTITY_EXTRACTION_MODE = "fast"
_config.GEMINI_EMBEDDING_MODEL = "test-model"
_config.GEMINI_GENERATION_MODEL = "test-model"
_config.BASE_DIR = str(BACKEND_DIR)
sys.modules["config"] = _config

# Create a mock gemini_client module
_gemini = types.ModuleType("gemini_client")
_rng = np.random.RandomState(42)

def _mock_generate_content(prompt, model=None):
    return "Mocked summary of the content."

def _mock_get_embedding(text):
    return _rng.randn(768).tolist()

def _mock_get_embeddings(texts):
    return [_rng.randn(768).tolist() for _ in texts]

def _mock_generate_document_summary(text, filename=""):
    return "This is a test document about various topics."

def _mock_generate_chunk_context(chunk, doc_summary, doc_name):
    return f"From '{doc_name}': test context."

def _mock_generate_with_image(image_data, prompt, mime_type="image/png"):
    return "Mocked image description."

_gemini.generate_content = _mock_generate_content
_gemini.get_embedding = _mock_get_embedding
_gemini.get_embeddings = _mock_get_embeddings
_gemini.generate_document_summary = _mock_generate_document_summary
_gemini.generate_chunk_context = _mock_generate_chunk_context
_gemini.generate_with_image = _mock_generate_with_image
_gemini.client = MagicMock()
_gemini.ContentType = MagicMock()
_gemini.PartType = MagicMock()
sys.modules["gemini_client"] = _gemini

# Also mock the embeddings module so it uses our mock
_embeddings = types.ModuleType("embeddings")
_embeddings.get_embeddings = _mock_get_embeddings
sys.modules["embeddings"] = _embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings():
    """10 synthetic embeddings in two clear clusters."""
    rng = np.random.RandomState(0)
    cluster_a = rng.randn(5, 50) + 3.0
    cluster_b = rng.randn(5, 50) - 3.0
    return np.vstack([cluster_a, cluster_b])


@pytest.fixture
def sample_entities_two_clusters():
    """Entity lists matching the two-cluster layout."""
    return (
        [
            [{"name": "Alpha", "type": "concept", "importance": 8}]
            for _ in range(5)
        ]
        + [
            [{"name": "Beta", "type": "concept", "importance": 7}]
            for _ in range(5)
        ]
    )


@pytest.fixture
def chromadb_collection(tmp_path):
    """Ephemeral ChromaDB collection backed by a temp dir."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma"),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    coll = client.get_or_create_collection("test_chunks")
    return coll
