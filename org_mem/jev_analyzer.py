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

"아무 계층도 필요 없음"은 분기에 넣지 않는다. 검색을 건너뛸지는 규칙
분석기가 정한다(`fallback_plan.memory_needed`).

## 검색 여부를 Jev에 묻지 않는 이유 (2026-10-19)

처음에는 `answer_source`(memory_prefetch / session_only / direct_answer)도
Jev에 물었다. 라우팅 80문항에서 **19건이 "검색 불필요"로 나왔고 80문항은
전부 검색이 필요한 질문이다.** 규칙 방어가 19건을 모두 되돌렸고, 반대로
Jev가 검색을 추가해 준 경우는 없었다. 즉 이 판정은 비용과 지연만 쓰고
최종 동작에 한 건도 반영되지 않았다.

원인은 모델이 아니라 **배치**다. 이 판정은 `state_for()`에 없는 정보를
요구한다.

    session_only  : 이전 **답변 본문**에 그 값이 있는지 알아야 한다.
                    state에는 이전 **질문**만 들어간다(본문은 일부러 뺐다).
    direct_answer : 조직 저장소에 그 내용이 있는지 알아야 한다.
                    state에는 저장소 목록이 없다.
    tier_branch   : 질문의 주제를 저장소 종류에 대응시키면 된다.
                    criteria가 그 대응을 적어 주므로 질문만으로 답이 된다.

실제로 `session_only` 9건은 대화 기록이 아예 없는 상태에서 골랐다 —
되짚을 대화가 존재하지 않는데 "되짚으면 되는 질문처럼 생겼다"는 문장
모양만 보고 답한 것이다. `direct_answer` 10건은 "TF와 IDF", "Naive RAG"
처럼 우리 MTM에 정리돼 있는 개념을 일반 지식으로 읽은 것이다.

고치는 방향은 두 가지였다. 설명 문구를 좁히거나(증상), 저장소 목록을
state에 넣거나(좁은 입력·빠른 결정이라는 Jev를 쓰는 이유를 깬다).
둘 다 택하지 않고 **질문에서 뺐다.** 검색 여부는 규칙이, 저장소 선택은
Jev가 맡는다. 질문이 3개에서 2개로 줄어 호출 입력도 작아진다.

## 실측 (2026-10-16)

    질문: "지난번에 얘기했던 그 일정, 공식 계획서랑 안 맞는 거 같은데 확인해줄래?"
    -> choice stm_ltm (0.77), confidence 0.73
       stm 0.01 · mtm 0 · ltm 0.13 · stm_mtm 0 · stm_ltm 0.77 · mtm_ltm 0.02 · all 0.07

    지연 0.65초 · 입력 580토큰 · $0.0000244
"""

from __future__ import annotations

import os
from typing import Any

from .jev_client import choice_of, noul_of
from .query_analyzer import QueryPlan, RuleBasedQueryAnalyzer


TIERS = ("stm", "mtm", "ltm")

#: `is_comparison`이 이 값을 넘으면 비교 질문으로 본다.
#: noul은 0~1 확률이고 confidence를 따로 주지 않으므로 이 값을 직접 쓴다.
COMPARISON_THRESHOLD = 0.5


def _widen_enabled() -> bool:
    """비교 질문일 때 계층을 하나 더 여는 보정. 기본 켜짐."""
    return os.environ.get("JEV_WIDEN_ON_COMPARISON", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }

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

#: 각 저장소에 **실제로 무엇이 들어 있는지**를 적는다.
#:
#: 첫 판에서는 계층을 일반적인 말("진행 중 문서", "공식 규정이나 매뉴얼")로
#: 설명했다. 그 결과 MTM 기대 29문항 중 12건이 LTM으로 갔다. 실패가 전부
#: 같은 모양이었다.
#:
#:     "라이브러리 RAG 문서에서 Ingestion과 Chunking 단계가 뭐야?"   -> ltm
#:     "MTEB-ko 테스트셋 규모가 어떻게 돼?"                          -> ltm
#:     "arctic-ko 라이센스 뭐였는지 얼마였어?"                        -> ltm
#:
#: 기술 조사 질문이 "문서·매뉴얼"로 읽혀서 LTM으로 간 것이다. 그런데 이
#: 코퍼스의 LTM 매뉴얼은 **연구비 집행 매뉴얼(RCMS)** 같은 행정 문서이고,
#: 기술 조사는 주차 회의자료(MTM)에 있다.
#:
#:     STM  세미나 피드백 10 · 킥오프 회의 1
#:     MTM  주차 회의자료 22   (기술 조사 · 개념 정리 · 벤치마크 수치 · 설계 변경)
#:     LTM  계획서 5 · 공식문서 9 · 매뉴얼 3   (예산 · 기간 · 협약 · 연구비 집행)
#:
#: 이것은 배경지식 주입이 아니라 **저장소 이름을 제대로 대는 것**이다.
#: 판별 모델에 긴 설명을 넣으면 좁은 결정이라는 장점이 사라지므로, 각 항목은
#: 저장소에 실제로 있는 자료 종류만 나열하고 추론 규칙은 적지 않는다.
BRANCH_CRITERIA: dict[str, str] = {
    "stm": (
        "세미나나 킥오프 회의에서 **구두로 받은 피드백·지적·제안**. "
        "누가 무엇을 하라고 했는지, 어떤 방향이 좋겠다고 했는지."
    ),
    "mtm": (
        "주차별 회의자료에 정리한 **기술 조사와 작업 기록**. "
        "검색 방식·임베딩·RAG 구조 같은 기술 개념 정리, 벤치마크 수치와 모델 비교, "
        "데이터셋 규모, 라이선스, 설계 변경 내역, 주차별 진행 상황."
    ),
    "ltm": (
        "과제 **행정 문서**. 연구개발계획서, 협약서, 과제번호·총사업비·연구개발기간, "
        "지원 조건, 연구비 집행 절차와 증빙 규정(RCMS), 특허·지표 같은 제출 항목."
    ),
    # 조합 분기는 "두 자료를 맞춰 봐야 하는 질문"의 모양을 적는다.
    # 첫 판에서는 "A와 B를 함께 봐야 한다"고만 적었고, 혼합 기대 12문항 중
    # 7건이 단일 분기로 떨어졌다. 어떤 질문이 두 곳을 필요로 하는지가
    # 설명에 없었다 — "지적을 받고 무엇을 했나", "목표 대비 지금 어디까지"처럼
    # 받은 쪽과 한 쪽이 다른 저장소에 있는 질문들이다.
    "stm_mtm": (
        "받은 피드백과 그에 따라 **실제로 한 작업**을 맞춰 봐야 한다. "
        "'지적을 받고 무엇을 바꿨나', '요청대로 무엇을 했나'처럼 "
        "말이 나온 곳과 반영된 곳이 서로 다른 질문."
    ),
    "stm_ltm": (
        "회의에서 나온 이야기와 **문서에 적힌 내용**이 맞는지 맞춰 봐야 한다. "
        "'그때 말한 것이 계획서에도 있나'처럼 구두 내용과 공식 문서를 대조하는 질문."
    ),
    "mtm_ltm": (
        "계획서에 **약속한 것**과 작업 기록의 **실제 결과**를 맞춰 봐야 한다. "
        "'목표 대비 지금 어디까지', '계획한 지표를 달성했나'처럼 "
        "기준이 행정 문서에 있고 실적이 작업 기록에 있는 질문."
    ),
    "all": "세 종류를 모두 봐야 한다",
}

#: 답을 어디서 가져오는가는 Jev에 묻지 않는다. 모듈 설명의
#: "검색 여부를 Jev에 묻지 않는 이유"를 참고한다. 규칙 분석기의
#: `answer_source`를 그대로 쓴다.

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
        comparison = noul_of(answers.get("is_comparison") or {})

        overrides: list[str] = []
        if branch not in BRANCH_TIERS:
            overrides.append(f"분기 {branch!r}가 목록에 없어 all로 둔다")
            branch = "all"

        # 검색 여부는 규칙이 정한다. Jev는 "검색한다면 어디를"만 답한다.
        source = fallback_plan.answer_source
        memory_needed = fallback_plan.memory_needed
        if memory_needed:
            tiers = BRANCH_TIERS[branch]
            tiers, widen_note = self._widen_if_comparison(
                tiers, comparison, branch_probabilities
            )
            if widen_note:
                overrides.append(widen_note)
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

    def _widen_if_comparison(
        self,
        tiers: tuple[str, ...],
        comparison: float,
        probabilities: dict[str, float],
    ) -> tuple[tuple[str, ...], str]:
        """비교 질문인데 계층을 하나만 골랐으면 한 계층을 더 연다.

        비교 질문은 **두 자료를 맞춰 보는 것**이라 한 저장소만 열면 비교 대상이
        없다. 라우팅 80문항에서 정답 계층이 둘인 12건이 여기 해당한다.

        넓히는 기준을 confidence로 두지 않는다. 실측에서 confidence는
        맞은 것(중앙 0.32)과 틀린 것(중앙 0.26)을 가르지 못했고, 임계값을
        0.4로 두면 틀린 14건을 건지는 대신 맞은 35건까지 함께 넓어졌다.
        대신 **질문의 뜻**을 쓴다 — `is_comparison`은 "두 자료의 차이·충돌을
        확인해 달라는 질문인가"를 따로 물은 값이다.

        더하는 계층은 분기 확률에서 **두 번째로 높은 분기**가 가리키는 것 중
        아직 열지 않은 계층이다. 임의로 `all`로 넓히지 않는다.

        `JEV_WIDEN_ON_COMPARISON=0`으로 끈다. 비교 실험에서 이 보정의
        효과만 따로 보려면 꺼야 한다.
        """
        if not _widen_enabled() or len(tiers) != 1 or comparison < COMPARISON_THRESHOLD:
            return tiers, ""
        ranked = sorted(probabilities.items(), key=lambda kv: -kv[1])
        for name, _ in ranked:
            for tier in BRANCH_TIERS.get(name, ()):
                if tier not in tiers:
                    widened = tuple(t for t in TIERS if t in (*tiers, tier))
                    return widened, (
                        f"비교 질문({comparison:.2f})이라 {'+'.join(tiers)} -> "
                        f"{'+'.join(widened)}로 넓힌다"
                    )
        return tiers, ""

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
