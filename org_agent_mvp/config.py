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
    # ---------------------------------------------- 근거 전달 예산 (2026-09-28)
    #
    # 세 값이 함께 "모델이 실제로 읽는 글자 수"를 정한다. 예전에는 셋 다
    # 코드에 박혀 있어서 재는 것이 불가능했다.
    #
    #   max_evidence_chars  근거 블록 전체 상한. 넘치면 뒷순위 카드가 빠진다.
    #   ltm_excerpt_chars   카드 한 장의 본문 길이. 청크 36.5%가 600자를 넘는다.
    #   ltm_chunks_per_doc  한 문서에서 몇 대목까지 보여 줄지.
    #
    # 기본값은 이전과 같다(7000 / 600 / 1). 올리면 답변 근거가 늘어나는 대신
    # 입력 토큰이 늘어난다. 바꿔서 재고 결정할 값이라 설정으로 뺐다.
    max_evidence_chars: int = 7000
    ltm_excerpt_chars: int = 600
    ltm_chunks_per_doc: int = 1
    #: 시드 문서(`memory_root/{stm,mtm,ltm}/`)의 카드 본문 길이.
    #:
    #: 0이면 `ltm_excerpt_chars`를 따른다(지금까지의 동작).
    #:
    #: 왜 나눴나 — 두 자료의 모양이 다르다. 청크 코퍼스는 한 카드가 청크
    #: 하나(평균 수백 자)이고 `ltm_chunks_per_doc`으로 대목을 늘릴 수 있다.
    #: 시드 문서는 **파일 하나가 카드 하나**라 그 손잡이가 걸리지 않는다.
    #:
    #: 2026-10-07 실측: 진행 과제 시드의 LTM 문서가 평균 21,582자인데 발췌가
    #: 600자면 문서의 **2.8%**만 전달된다. 답을 아는 20문항에서 문서는 20/20
    #: 찾았는데 전달된 글자에 답이 있는 것은 16/20이었다. 6,000자로 올리면
    #: 18/20이 된다. 한 값을 공유하면 Allganize 청크 설정까지 같이 움직여서
    #: 두 측정을 섞게 되므로 손잡이를 나눴다.
    seed_excerpt_chars: int = 0
    # ---------------------------------------- 도구 결과 예산 (2026-09-29)
    #
    # 위 세 값은 **1차 컨텍스트**를 정한다. retrieve_memory 결과는 같은
    # 검색 경로를 타므로 같은 예산이 걸렸는데, 그게 비용의 절반을 먹었다.
    #
    # 실측(20문항): 1차 컨텍스트를 3배로 키웠더니 도구 결과도 2.4배가 됐고,
    # 재검색이 있는 문항은 그것이 호출 수만큼 다시 실려 입력 토큰의 46%를
    # 차지했다. 1차 컨텍스트는 카드 8장이지만 도구 결과는 10~20장이라
    # 같은 깊이를 주면 훨씬 비싸다.
    #
    # 그래서 도구 결과는 따로 잡는다. 기본값은 예전 고정값(600자 / 1대목)이다.
    tool_result_excerpt_chars: int = 600
    tool_result_chunks_per_doc: int = 1
    #: 도구 결과에서 **이미 전달한 문서**를 빼고 뒤 순위로 채운다.
    #:
    #: 2026-09-29 실측(20문항): 재검색이 가져온 카드 96장 중 66장(69%)이
    #: 이미 1차 컨텍스트나 앞선 도구 호출로 전달한 문서였다. 한 문항(q_180)은
    #: 5장 전부가 중복이었다. 같은 문서가 컨텍스트에 두 번 쌓이고, 재검색이
    #: 있으면 호출마다 다시 실린다.
    #:
    #: 이미 프롬프트에 있는 내용이므로 빼도 모델이 잃는 정보가 없다.
    #: 기본은 꺼짐이다 — 켜고 재서 효과를 확인한 뒤 판단한다.
    tool_result_dedupe: bool = False
    #: 모델이 요청할 수 있는 retrieve_memory top_k의 상한. 0이면 제한하지 않는다.
    #:
    #: 모델은 top_k=20까지 요청한다(58회 중 3회). 1차 컨텍스트가 8장인데 도구
    #: 결과가 20장이면 그쪽이 더 무겁고, 그 2문항이 가장 비싼 문항이었다.
    tool_result_top_k_cap: int = 0
    # ------------------------------------------ 세션 전달 원장 (2026-10-06)
    #
    # 지금까지의 중복 제거는 **한 턴 안**에서만 돌았다. `seen_tier_queries`와
    # `AgentTrace`가 턴마다 새로 만들어지기 때문이다. 그래서 턴이 넘어가면
    # prefetch가 백지에서 같은 문서를 다시 가져오고, 같은 본문이 또 실린다.
    #
    # 원장을 켜면 이미 본문을 전달한 문서는 다음 턴에 **본문 없이 참조 카드**로
    # 들어간다(제목 · 출처 · evidence_id · 몇 번째 턴에 줬는지). 본문이 다시
    # 필요하면 모델이 expand_evidence로 되불러온다 — 숨기는 게 아니라 줄이는 것이다.
    #
    # 숨기면 안 되는 이유는 09-29에 이미 봤다. q_180은 도구 결과 5장이 전부
    # 중복이라 빈 목록이 나갔고, 모델은 "찾아도 없다"로 읽어 한 번 더 검색했다.
    #
    # 기본은 꺼짐이다. 멀티턴 측정 경로가 아직 없어 효과를 재지 않았다
    # (2026-10-06 기준 세션 로그 201개가 전부 1턴이다).
    session_evidence_ledger: bool = False
    #: 원장에 남길 문서 수 상한. 넘치면 오래된 턴의 것부터 버린다. 0이면 무제한.
    session_ledger_max_docs: int = 200
    #: 원장을 켜도 한 턴에 본문을 최소 몇 장은 보낸다.
    #:
    #: 새로 볼 문서가 없으면 후보 전부가 참조 카드가 되는데, 이전 턴의 본문은
    #: 메시지에 남아 있지 않다(턴마다 messages를 새로 만든다. 남는 것은
    #: answer_summary 500자뿐이다). 그 상태로는 "근거 없음"으로 답하거나
    #: expand_evidence를 불러 왕복이 는다.
    #:
    #: 그래서 새 문서가 이 값보다 적으면 상위 참조를 본문으로 되돌린다.
    session_ledger_min_bodies: int = 2

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
            max_evidence_chars=int(os.environ.get("CONTEXT_MAX_EVIDENCE_CHARS", "7000")),
            ltm_excerpt_chars=int(os.environ.get("LTM_EXCERPT_CHARS", "600")),
            ltm_chunks_per_doc=int(os.environ.get("LTM_CHUNKS_PER_DOC", "1")),
            seed_excerpt_chars=int(os.environ.get("SEED_EXCERPT_CHARS", "0")),
            tool_result_excerpt_chars=int(
                os.environ.get("TOOL_RESULT_EXCERPT_CHARS", "600")
            ),
            tool_result_chunks_per_doc=int(
                os.environ.get("TOOL_RESULT_CHUNKS_PER_DOC", "1")
            ),
            tool_result_dedupe=os.environ.get("TOOL_RESULT_DEDUPE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            tool_result_top_k_cap=int(os.environ.get("TOOL_RESULT_TOP_K_CAP", "0")),
            session_evidence_ledger=os.environ.get("SESSION_EVIDENCE_LEDGER", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            session_ledger_max_docs=int(
                os.environ.get("SESSION_LEDGER_MAX_DOCS", "200")
            ),
            session_ledger_min_bodies=int(
                os.environ.get("SESSION_LEDGER_MIN_BODIES", "2")
            ),
        )
