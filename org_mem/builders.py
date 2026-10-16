"""실행·평가가 공유하는 구성 함수.

## 왜 모았나

같은 구성을 네 군데서 각자 조립하고 있었다.

    org_mem/__main__.py                   CLI
    eval/run_eval.py                      평가 러너
    scripts/bench_prentice_routing.py     라우팅 측정
    scripts/check_prentice_delivery.py    전달 측정

그래서 한 곳을 고쳐도 다른 곳은 옛 설정으로 돌았다. 실제로 생긴 문제가
두 가지다.

1. **측정 스크립트 전부가 `RuleBasedQueryAnalyzer`를 하드코딩**하고 있었다.
   그래서 `LLMQueryAnalyzer`는 한 번도 측정된 적이 없다. 라우팅 75/80,
   연쇄 26/27, 전달 20/20은 모두 규칙 기반 결과다. 분석기를 바꿔 끼우려면
   네 파일을 각각 고쳐야 했다.
2. 실제 런타임은 원장 설정(`ledger_cap_ratio`·`ledger_min_bodies`)을
   넘기는데 일부 평가 구성은 넘기지 않았다. 지금은 기본값이 같아 결과가
   같지만, 기본값을 바꾸면 의도와 다르게 평가된다.

여기서는 **어느 분석기를 쓸지만 한 글자로 고르게** 한다. 평가 목적에 따라
달라야 하는 값(비교군 파라미터)은 호출부가 명시적으로 넘긴다.

## 분석기 고르기

    QUERY_ANALYZER=rule     신호어 규칙 (기본, API 호출 없음)
    QUERY_ANALYZER=llm      작은 LLM이 JSON 계획을 생성
    QUERY_ANALYZER=jev      Jev가 7분기 중 하나를 골라 그 저장소만 검색

`rule`은 API 키가 없어도 돌아간다. 나머지는 키가 필요하다.
"""

from __future__ import annotations

import os
from typing import Any

from .config import AppConfig
from .context_builder import ContextBuilder
from .memory_store import build_memory_store
from .mock_llm import MockLLMClient
from .prefetch import MemoryPrefetcher


ANALYZER_KINDS = ("rule", "llm", "jev")
DEFAULT_ANALYZER = "rule"


def analyzer_kind(explicit: str | None = None) -> str:
    """쓸 분석기 이름. 인자가 있으면 환경변수보다 우선한다."""
    name = (explicit or os.environ.get("QUERY_ANALYZER", DEFAULT_ANALYZER)).strip().lower()
    return name if name in ANALYZER_KINDS else DEFAULT_ANALYZER


def build_query_analyzer(
    config: AppConfig,
    memory_store: Any,
    *,
    kind: str | None = None,
    client: Any = None,
) -> Any:
    """`analyze(query, recent_turns) -> QueryPlan`을 지키는 분석기를 만든다.

    `client`를 주면 그것을 쓴다(테스트·mock용). 없으면 종류에 맞는 것을 만든다.
    `rule`은 클라이언트를 쓰지 않는다.

    API 키가 없는데 `llm`/`jev`를 고르면 조용히 규칙으로 내려가지 않고
    예외를 던진다. 조용한 강등은 "Jev를 측정했다"고 착각하게 만든다.
    """
    vocabulary = memory_store.filter_vocabulary()
    chosen = analyzer_kind(kind)

    if chosen == "rule" or isinstance(client, MockLLMClient):
        from .query_analyzer import RuleBasedQueryAnalyzer

        return RuleBasedQueryAnalyzer(vocabulary=vocabulary)

    if not config.api_key:
        raise SystemExit(
            f"QUERY_ANALYZER={chosen}인데 OPENROUTER_API_KEY가 비어 있다. "
            "키를 채우거나 QUERY_ANALYZER=rule로 실행한다."
        )

    if chosen == "jev":
        from .jev_analyzer import JevQueryAnalyzer
        from .jev_client import JevClient, model_name

        return JevQueryAnalyzer(
            JevClient(config, model=model_name()),
            model=model_name(),
            vocabulary=vocabulary,
        )

    from .openrouter_client import OpenRouterClient
    from .query_analyzer import LLMQueryAnalyzer

    return LLMQueryAnalyzer(
        client or OpenRouterClient(config),
        model=config.query_analyzer_model,
        max_tokens=config.query_analyzer_max_tokens,
        # 코퍼스에 실제로 있는 값만 enum으로 열어 준다.
        vocabulary=vocabulary,
    )


def build_prefetcher(
    config: AppConfig,
    memory_store: Any,
    *,
    total_top_k: int | None = None,
    **overrides: Any,
) -> MemoryPrefetcher:
    """런타임과 같은 설정으로 prefetcher를 만든다.

    예전에는 평가 쪽이 원장 인자를 빼고 만들어서, 기본값이 바뀌면 조용히
    다른 것을 재게 되어 있었다. 여기서는 전부 설정에서 가져오고,
    비교군만 `overrides`로 덮어쓴다.
    """
    kwargs: dict[str, Any] = {
        "alpha": config.tier_prior_alpha,
        "pool_per_tier": config.prefetch_pool_per_tier,
        "cut_ratio": config.prefetch_cut_ratio,
        "min_cards": config.prefetch_min_cards,
        "tier_floor": config.prefetch_tier_floor,
        "ledger_cap_ratio": config.session_ledger_cap_ratio,
    }
    kwargs.update(overrides)
    return MemoryPrefetcher(
        memory_store,
        total_top_k if total_top_k is not None else config.prefetch_top_k,
        **kwargs,
    )


def build_context_builder(
    config: AppConfig,
    *,
    context_mode: str | None = None,
    **overrides: Any,
) -> ContextBuilder:
    kwargs: dict[str, Any] = {
        "context_mode": context_mode or config.context_mode,
        "max_evidence_chars": config.max_evidence_chars,
        "ledger_min_bodies": config.session_ledger_min_bodies,
    }
    kwargs.update(overrides)
    return ContextBuilder(**kwargs)


def build_runtime(config: AppConfig, use_mock: bool, *, analyzer: str | None = None):
    """CLI와 평가가 함께 쓰는 런타임 조립.

    import를 함수 안에서 하는 것은 순환 참조를 피하려는 것이다
    (`agent_runtime`이 이 모듈을 쓰지는 않지만, 설정 모듈끼리 엮이지 않게 둔다).
    """
    from .agent_runtime import AgentRuntime
    from .openrouter_client import OpenRouterClient

    client = MockLLMClient() if use_mock else OpenRouterClient(config)
    memory_store = build_memory_store(config)
    return AgentRuntime(
        config=config,
        client=client,
        memory_store=memory_store,
        query_analyzer=build_query_analyzer(
            config, memory_store, kind=analyzer, client=client
        ),
    )


def describe(config: AppConfig, analyzer: Any) -> dict[str, Any]:
    """이번 실행의 유효 설정. 로그·결과 파일 머리에 붙인다.

    설정이 여러 모듈에 흩어져 있어서, 결과만 보고 "무슨 설정으로 돌았나"를
    되읽을 수 없었다. API 키는 넣지 않는다.
    """
    from . import rerank, scoring

    return {
        "analyzer": type(analyzer).__name__,
        "analyzer_model": getattr(analyzer, "model", ""),
        "agent_model": config.agent_model,
        "scorer": scoring.scorer_name(),
        "scorer_seed": scoring.scorer_name("seed"),
        "bm25_b": scoring.bm25_b(),
        "bm25_b_seed": scoring.bm25_b("seed"),
        "normalize": scoring.normalize_mode(),
        "rerank": rerank.model_name() or "none",
        "prefetch_top_k": config.prefetch_top_k,
        "pool_per_tier": config.prefetch_pool_per_tier,
        "cut_ratio": config.prefetch_cut_ratio,
        "tier_floor": config.prefetch_tier_floor,
        "tier_prior_alpha": config.tier_prior_alpha,
        "context_mode": config.context_mode,
        "max_evidence_chars": config.max_evidence_chars,
        "tool_result_dedupe": config.tool_result_dedupe,
        "session_evidence_ledger": config.session_evidence_ledger,
    }
