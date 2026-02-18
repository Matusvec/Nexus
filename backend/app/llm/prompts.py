"""Prompt registry — loads and manages versioned prompt templates."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

PROMPT_REGISTRY: dict[str, dict[str, str]] = {}  # {name: {version: filepath}}
_LATEST: dict[str, str] = {}  # {name: latest_version}


def _discover_prompts() -> None:
    """Scan prompts/ directory and build registry."""
    if not PROMPTS_DIR.exists():
        logger.warning("Prompts directory not found: %s", PROMPTS_DIR)
        return
    for f in sorted(PROMPTS_DIR.glob("*.txt")):
        # Parse filename: extract_problems_v1.0.txt → name=extract_problems, version=v1.0
        stem = f.stem  # e.g., "extract_problems_v1.0"
        parts = stem.rsplit("_v", 1)
        if len(parts) != 2:
            logger.warning("Skipping unrecognized prompt file: %s", f.name)
            continue
        name = parts[0]
        version = f"v{parts[1]}"
        PROMPT_REGISTRY.setdefault(name, {})[version] = str(f)
        _LATEST[name] = version  # last sorted = latest version


_discover_prompts()


def get_prompt(name: str, version: str | None = None) -> str:
    """Load prompt template by name and optional version. Defaults to latest."""
    versions = PROMPT_REGISTRY.get(name)
    if not versions:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(PROMPT_REGISTRY.keys())}")
    target_version = version or _LATEST.get(name)
    filepath = versions.get(target_version)
    if not filepath:
        raise ValueError(f"Version {target_version} not found for prompt {name}. Available: {list(versions.keys())}")
    return Path(filepath).read_text(encoding="utf-8")


def get_prompt_version(name: str) -> str:
    """Get the latest version string for a prompt name."""
    return _LATEST.get(name, "unknown")


def list_prompts() -> dict[str, list[str]]:
    """Return all registered prompts and their versions."""
    return {name: sorted(versions.keys()) for name, versions in PROMPT_REGISTRY.items()}
