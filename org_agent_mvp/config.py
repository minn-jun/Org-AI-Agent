from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    memory_root: Path
    api_key: str
    model: str
    base_url: str
    app_name: str
    site_url: str
    max_tool_calls: int = 3
    max_same_tier_calls: int = 1
    default_top_k: int = 5
    prefetch_top_k: int = 8
    session_cache_turns: int = 8

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env")
        return cls(
            project_root=PROJECT_ROOT,
            memory_root=PROJECT_ROOT / "memory_seed",
            api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
            model=os.environ.get(
                "OPENROUTER_MODEL", "google/gemma-4-31b-it:free"
            ).strip(),
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            app_name=os.environ.get("OPENROUTER_APP_NAME", "Org Agent MVP"),
            site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            prefetch_top_k=int(os.environ.get("PREFETCH_TOP_K", "8")),
            session_cache_turns=int(os.environ.get("SESSION_CACHE_TURNS", "8")),
        )
