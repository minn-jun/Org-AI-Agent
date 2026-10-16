"""Jev로 계층 분기를 판정하는 질의 분석기.

규칙 기반·LLM 기반이 하던 일 중 **고정 선택지 판정만** Jev로 옮긴다.
나머지(검색어 조립·과제명 추출·정합성 보정)는 코드가 그대로 맡는다.

## 가중치 곱이 아니라 저장소 선택이다

기존 두 분석기는 `memory_weights`를 **순위 보정값**으로 썼다. 세 계층을
모두 검색한 뒤 `final_score = 정규화점수 × (1 + alpha × prior)`로 곱했다.
계층은 "조금 더 본다 / 조금 덜 본다"였고 어느 계층도 빠지지 않았다.
규칙 기반이 공식 문서 질문에도 stm 0.10을 주는 것이 그 증거다.

이 분석기는 다르다. 7분기 중 확률이 가장 높은 하나를 고르고,
**고른 분기에 들어간 계층만 검색한다.**

    규칙:  stm 0.10 · mtm 0.20 · ltm 0.70  -> 세 계층 검색, 점수에 곱
    Jev:   stm 0.00 · mtm 0.00 · ltm 1.00  -> LTM만 검색, 곱은 균일

고른 계층은 모두 1.0이라 곱셈 보정이 균일해져(1 + alpha) 순위를 흔들지
않는다. 즉 가중치는 "검색할지 말지"의 켜고 끄기로만 쓰인다.
`MemoryPrefetcher`는 가중치 0인 계층을 검색하지 않고 자리도 주지 않는다.

## 왜 3택이 아니라 7택인가

계층은 배타적이지 않다. 진행 과제 라우팅 80문항 중 **12건이 정답 계층이
둘**이다(ltm+mtm 6 · mtm+stm 5 · ltm+stm 1). 하나만 고르게 하면 이 12건은
구조적으로 틀린다. 그래서 조합까지 포함한 7택으로 묻는다.

"아무 계층도 필요 없음"은 분기에 넣지 않는다. 검색을 건너뛰는 결정은
`answer_source`가 따로 맡고, 아래 규칙 방어가 한 번 더 막는다.

## 실측 (2026-10-16)

    질문: "지난번에 얘기했던 그 일정, 공식 계획서랑 안 맞는 거 같은데 확인해줄래?"
    -> choice stm_ltm (0.77), confidence 0.73
       stm 0.01 · mtm 0 · ltm 0.13 · stm_mtm 0 · stm_ltm 0.77 · mtm_ltm 0.02 · all 0.07

    지연 0.65초 · 입력 580토큰 · $0.0000244
"""

from __future__ import annotations

from typing import Any

from .jev_client import choice_of, noul_of
from .query_analyzer import QueryPlan, RuleBasedQueryAnalyzer


TIERS = ("stm", "mtm", "ltm")

#: 분기 이름 -> 실제로 검색할 계층.
BRANCH_TIERS: dict[str, tuple[str, ...]] = {
    "stm": ("stm",),
    "mtm": ("mtm",),
    "ltm": ("ltm",),
    "stm_mtm": ("stm", "mtm"),
    "stm_ltm": ("stm", "ltm"),
    "mtm_ltm": ("mtm", "ltm"),
    "all": ("stm", "mtm", "ltm"),
}

#: Jev에 보내는 분기 설명. 선택지 이름만으로 뜻이 서게 적는다.
#: 긴 배경지식을 넣으면 판별 모델의 장점(좁은 결정)이 사라진다.
BRANCH_CRITERIA: dict[str, str] = {
    "stm": "최근 대화·오늘이나 방금 정해진 것·세미나 피드백 같은 단기 기록만 보면 된다",
    "mtm": "주간 회의 자료·초안·진행 중인 보고서 같은 작업 중 문서만 보면 된다",
    "ltm": "공식 계획서·협약서·승인된 문서·조직 공통 규정이나 매뉴얼만 보면 된다",
    "stm_mtm": "단기 기록과 진행 중 문서를 함께 봐야 한다",
    "stm_ltm": "단기 기록과 공식 문서를 함께 봐야 한다",
    "mtm_ltm": "진행 중 문서와 공식 문서를 함께 봐야 한다",
    "all": "세 종류를 모두 봐야 한다",
}

#: 답을 어디서 가져오는가.
SOURCE_CRITERIA: dict[str, str] = {
    "memory_prefetch": "조직에 쌓인 문서나 기록을 찾아봐야 답할 수 있다",
    "session_only": "바로 앞 대화에 이미 나온 내용만 되짚으면 답이 된다",
    "direct_answer": "일반 상식이나 개념 설명이라 조직 자료가 필요 없다",
}

#: 단일 계층 분기 -> intent. 조합 분기는 organization_memory_lookup으로 둔다.
_BRANCH_INTENT = {
    "stm": "recent_context_lookup",
    "mtm": "working_document_lookup",
    "ltm": "official_knowledge_lookup",
}

QUESTIONS: dict[str, dict[str, Any]] = {
    "tier_branch": {
        "type": "choice",
        "instructions": (
            "이 질문에 답하려면 어느 저장소를 찾아봐야 하는가? "
            "질문이 가리키는 자료의 종류로 판단한다."
        ),
        "criteria": BRANCH_CRITERIA,
    },
    "answer_source": {
        "type": "choice",
        "instructions": "이 질문의 답은 어디에서 나오는가?",
        "criteria": SOURCE_CRITERIA,
    },
    "is_comparison": {
        "type": "noul",
        "instructions": "이 질문이 두 자료의 차이·충돌·일치 여부를 확인해 달라는 것인가?",
    },
}


class JevQueryAnalyzer:
    """Jev로 계층 분기·답변 출처를 판정하고, 나머지는 규칙에서 가져온다.

    `QueryAnalyzer` 계약(`analyze(query, recent_turns) -> QueryPlan`)을 지키므로
    런타임과 평가 스크립트의 구조를 바꾸지 않고 바꿔 끼울 수 있다.

    호출 실패·형식 오류는 규칙 기반 결과로 되돌린다. 판정 모델이 죽었을 때
    검색이 멈추는 것이 가장 나쁜 실패다.
    """

    def __init__(
        self,
        client: Any,
        model: str | None = None,
        fallback: RuleBasedQueryAnalyzer | None = None,
        vocabulary: dict[str, list[str]] | None = None,
    ):
        self.client = client
        self.model = model or getattr(client, "model", "")
        self.vocabulary = vocabulary or {}
        # fallback에도 같은 어휘를 넘긴다. 넘기지 않으면 "이 과제" 같은 오탐이
        # 하드 필터로 들어간다(LLMQueryAnalyzer에서 실제로 겪은 문제다).
        self.fallback = fallback or RuleBasedQueryAnalyzer(vocabulary=self.vocabulary)
        self.last_usage: dict[str, Any] = {}
        self.last_fallback_used: bool = False
        #: 이번 판정의 전말. Jev 원래 답 / 규칙 답 / 최종 적용 / 덮어쓴 이유.
        #: 규칙 방어가 Jev 판단을 덮어쓰므로, 이것을 나눠 적지 않으면
        #: "Jev가 무엇을 골랐는지"를 결과에서 되읽을 수 없다.
        self.last_decision: dict[str, Any] = {}

    def analyze(
        self,
        query: str,
        recent_turns: list[dict[str, Any]] | None = None,
    ) -> QueryPlan:
        recent_turns = recent_turns or []
        self.last_usage = {}
        self.last_fallback_used = False
        self.last_decision = {}
        fallback_plan = self.fallback.analyze(query, recent_turns)
        try:
            response = self.client.decide(
                self.state_for(query, recent_turns),
                QUESTIONS,
                model=self.model or None,
            )
        except Exception as exc:
            self.last_fallback_used = True
            self.last_decision = {"error": f"{type(exc).__name__}: {exc}"}
            return QueryPlan(
                **{
                    **fallback_plan.to_dict(),
                    "reason": f"{fallback_plan.reason} (Jev 호출 실패로 규칙 사용: {exc})",
                }
            )
        self.last_usage = response.get("usage") or {}
        return self._plan_from(response, query, recent_turns, fallback_plan)

    def state_for(
        self,
        query: str,
        recent_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """판정에 넣는 상태.

        LLM 분석기(`LLMQueryAnalyzer._call_llm`)와 **같은 정보만** 넣는다.
        최근 4턴의 질문·의도·과제·출처 ID다. 이전 답변 본문은 양쪽 모두
        넣지 않는다. 한쪽에만 더 주면 모델 비교가 아니라 입력량 비교가 된다.
        """
        compact = [
            {
                "질문": str(turn.get("user_query", ""))[:200],
                "의도": str(turn.get("query_intent", "")),
                "과제": str(
                    turn.get("query_analysis", {}).get("filters", {}).get("project", "")
                ),
                "출처": list(turn.get("source_ids", []))[:5],
            }
            for turn in recent_turns[-4:]
        ]
        state: dict[str, Any] = {"사용자_질문": query.strip()}
        if compact:
            state["최근_대화"] = compact
        return state

    def _plan_from(
        self,
        response: dict[str, Any],
        query: str,
        recent_turns: list[dict[str, Any]],
        fallback_plan: QueryPlan,
    ) -> QueryPlan:
        answers = response.get("answers") or {}
        branch, branch_probabilities, branch_confidence = choice_of(
            answers.get("tier_branch") or {}
        )
        jev_source, source_probabilities, source_confidence = choice_of(
            answers.get("answer_source") or {}
        )
        comparison = noul_of(answers.get("is_comparison") or {})

        overrides: list[str] = []
        if branch not in BRANCH_TIERS:
            overrides.append(f"분기 {branch!r}가 목록에 없어 all로 둔다")
            branch = "all"
        source = jev_source
        if source not in SOURCE_CRITERIA:
            overrides.append(f"출처 {source!r}가 목록에 없어 memory_prefetch로 둔다")
            source = "memory_prefetch"

        # 규칙이 검색이 필요하다고 보면 Jev가 불필요라 해도 검색을 유지한다.
        #
        # 불필요한 검색은 토큰을 더 쓸 뿐이지만, 필요한 검색을 건너뛰면 근거 없이
        # 답하게 되어 실패 방향이 훨씬 나쁘다. 다만 이 방어가 켜져 있는 동안에는
        # "Jev가 불필요한 검색을 줄였다"를 관측할 수 없다. 그래서 덮어쓴 사실을
        # last_decision에 남겨 나중에 비율을 셀 수 있게 한다.
        if source != "memory_prefetch" and fallback_plan.memory_needed:
            overrides.append(
                f"규칙은 검색이 필요하다고 보아 {source} -> memory_prefetch로 되돌린다"
            )
            source = "memory_prefetch"

        memory_needed = source == "memory_prefetch"
        if memory_needed:
            tiers = BRANCH_TIERS[branch]
            # 고른 계층만 1.0. 균일하므로 (1 + alpha × prior) 곱이 순위를
            # 바꾸지 않는다. 0인 계층은 검색도 자리 배정도 받지 않는다.
            weights = {tier: (1.0 if tier in tiers else 0.0) for tier in TIERS}
        else:
            tiers = ()
            weights = {tier: 0.0 for tier in TIERS}

        intent = self._intent_for(source, branch, comparison)
        rewrites = list(fallback_plan.query_rewrites)
        if query.strip() not in rewrites:
            rewrites.insert(0, query.strip())

        self.last_decision = {
            "model": str(response.get("model") or self.model),
            "latency_s": response.get("latency_s"),
            "cost_usd": (response.get("usage") or {}).get("cost_usd"),
            "jev": {
                "tier_branch": branch,
                "branch_probabilities": branch_probabilities,
                "branch_confidence": branch_confidence,
                "answer_source": jev_source,
                "source_probabilities": source_probabilities,
                "source_confidence": source_confidence,
                "is_comparison": comparison,
            },
            "rule": {
                "memory_needed": fallback_plan.memory_needed,
                "answer_source": fallback_plan.answer_source,
                "memory_weights": fallback_plan.memory_weights,
                "intent": fallback_plan.intent,
            },
            "applied": {
                "answer_source": source,
                "searched_tiers": list(tiers),
                "memory_weights": weights,
                "intent": intent,
            },
            "overrides": overrides,
        }

        branch_label = "+".join(tiers) if tiers else "검색 없음"
        return QueryPlan(
            intent=intent,
            can_answer_directly=not memory_needed,
            memory_needed=memory_needed,
            answer_source=source,
            use_session_context=bool(recent_turns)
            and (source == "session_only" or "stm" in tiers),
            memory_weights=weights,
            query_rewrites=list(dict.fromkeys(rewrites)),
            # 과제명·문서 유형은 규칙에서 가져온다. 코퍼스의 과제가 1건이라
            # Jev에 물어도 선택지가 사실상 없고, 규칙 쪽에 이미 어휘 검증이 있다.
            filters=dict(fallback_plan.filters),
            reason=(
                f"Jev 판정: {branch} ({branch_confidence:.2f}) -> {branch_label}"
                + (f" / 보정: {'; '.join(overrides)}" if overrides else "")
            ),
        )

    def _intent_for(self, source: str, branch: str, comparison: float) -> str:
        """intent는 판정에서 결정적으로 따라온다. 따로 묻지 않는다.

        `memory_weights`가 저장소 선택으로 바뀌었어도 intent는 남겨 둔다.
        기존 로그·평가 스크립트가 이 값으로 집계한다.
        """
        if source == "session_only":
            return "session_context_answer"
        if source == "direct_answer":
            return "direct_answer"
        if comparison >= 0.5:
            return "memory_comparison"
        return _BRANCH_INTENT.get(branch, "organization_memory_lookup")
