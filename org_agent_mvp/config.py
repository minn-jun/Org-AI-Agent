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
    #: 실코퍼스를 LTM으로 붙일 때의 chunks.jsonl 경로. None이면 `ltm/` 폴더만 쓴다.
    ltm_corpus_path: Path | None
    api_key: str
    query_analyzer_model: str
    agent_model: str
    base_url: str
    app_name: str
    site_url: str
    max_tool_calls: int = 3
    default_top_k: int = 5
    prefetch_top_k: int = 8
    session_cache_turns: int = 8
    # A방식: tier 가중치를 검색 자리 수가 아니라 점수 prior로 사용한다.
    tier_prior_alpha: float = 1.0
    prefetch_pool_per_tier: int = 20
    prefetch_cut_ratio: float = 0.3
    prefetch_min_cards: int = 1
    prefetch_tier_floor: int = 1
    # 필터가 어긋난 문서에 곱하는 감점. 0이면 하드 필터, 1이면 필터 무시
    filter_penalty: float = 0.3
    query_analyzer_max_tokens: int = 4096
    # B방식: 1차 컨텍스트에 원문을 얼마나 넣을지 (full | summary | hybrid)
    context_mode: str = "full"

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env")
        agent_model = os.environ.get(
            "AGENT_MODEL",
            os.environ.get(
                "OPENROUTER_MODEL",
                "nvidia/nemotron-3-super-120b-a12b:free",
            ),
        ).strip()
        # 시드 세트를 바꿔 끼울 수 있게 열어 둔다. 상대 경로면 프로젝트 루트 기준이다.
        # 기본값 memory_seed_20200504는 실제 과제 기반 STM/MTM이라 저장소에 없다.
        # 없는 폴더를 주면 문서 0건으로 뜬다. 테스트는 tests/fixtures/memory를 직접 쓴다.
        memory_root = Path(os.environ.get("MEMORY_ROOT", "memory_seed_20200504"))
        if not memory_root.is_absolute():
            memory_root = PROJECT_ROOT / memory_root

        # LTM_CORPUS를 주면 실제 과제 문서를 LTM으로 쓴다.
        # "auto"면 저장소가 나란히 놓인 기본 배치에서 찾아본다.
        raw_corpus = os.environ.get("LTM_CORPUS", "").strip()
        ltm_corpus_path: Path | None = None
        if raw_corpus:
            if raw_corpus.lower() == "auto":
                from .ltm_corpus import default_corpus_path

                candidate = default_corpus_path(PROJECT_ROOT)
            else:
                candidate = Path(raw_corpus)
                if not candidate.is_absolute():
                    candidate = PROJECT_ROOT / candidate
            ltm_corpus_path = candidate if candidate.exists() else None

        return cls(
            project_root=PROJECT_ROOT,
            memory_root=memory_root,
            ltm_corpus_path=ltm_corpus_path,
            api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
            query_analyzer_model=os.environ.get(
                "QUERY_ANALYZER_MODEL",
                "liquid/lfm-2.5-2.6b:free",
            ).strip(),
            agent_model=agent_model,
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).rstrip("/"),
            app_name=os.environ.get("OPENROUTER_APP_NAME", "Org Agent MVP"),
            site_url=os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
            prefetch_top_k=int(os.environ.get("PREFETCH_TOP_K", "8")),
            session_cache_turns=int(os.environ.get("SESSION_CACHE_TURNS", "8")),
            tier_prior_alpha=float(os.environ.get("TIER_PRIOR_ALPHA", "1.0")),
            prefetch_pool_per_tier=int(os.environ.get("PREFETCH_POOL_PER_TIER", "20")),
            prefetch_cut_ratio=float(os.environ.get("PREFETCH_CUT_RATIO", "0.3")),
            prefetch_min_cards=int(os.environ.get("PREFETCH_MIN_CARDS", "1")),
            prefetch_tier_floor=int(os.environ.get("PREFETCH_TIER_FLOOR", "1")),
            filter_penalty=float(os.environ.get("FILTER_PENALTY", "0.3")),
            query_analyzer_max_tokens=int(
                os.environ.get("QUERY_ANALYZER_MAX_TOKENS", "4096")
            ),
            context_mode=os.environ.get("CONTEXT_MODE", "full").strip().lower(),
        )
