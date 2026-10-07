from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .query_analyzer import QueryPlan


#: 1차 컨텍스트에 원문을 얼마나 넣을지 정한다.
#: full   - 근거 원문을 전부 넣는다 (비교 기준선)
#: summary- 제목과 요약만 넣고 원문은 expand_evidence로 요청받는다
#: hybrid - 상위 몇 건만 원문을 넣고 나머지는 요약만 넣는다
CONTEXT_MODES = ("full", "summary", "hybrid")
HYBRID_FULL_CARDS = 2


@dataclass(frozen=True)
class BuiltContext:
    """만들어진 컨텍스트와, 그 안에 **실제로 들어간** 근거 목록.

    이 클래스가 있는 이유는 기록 때문이다. 이전에는 `build()`가 문자열만
    돌려줘서, 런타임은 글자 수 제한에 걸려 빠진 카드를 알 방법이 없었다.
    그래서 근거 개수와 final_sources에 **검색으로 찾은 후보 전체**를 적었다.
    실측(2026-09-28, Allganize 20문항)에서 20건 중 15건이 후보와 주입이
    달랐다 — 후보 8건 중 6~7건만 들어갔다. 그 상태로는 "근거를 줬는데
    모델이 못 썼다"와 "근거가 애초에 안 들어갔다"를 구분할 수 없다.
    """

    text: str
    #: 프롬프트에 실제로 들어간 카드(요약 형태). 순서는 주입 순서다.
    injected: list[dict[str, Any]] = field(default_factory=list)
    #: 앞선 턴에서 이미 본문을 전달해 **참조만** 넣은 카드.
    #:
    #: `injected`와 나눠 둔다. 참조 카드는 제목과 출처만 있고 본문이 없어서,
    #: 같은 칸에 적으면 "이번 턴에 본문을 전달했다"로 읽힌다. 기록을 나눈
    #: 이유와 같은 이유다.
    reference: list[dict[str, Any]] = field(default_factory=list)
    #: 글자 수 제한에 걸려 빠진 카드의 evidence_id.
    dropped_evidence_ids: list[str] = field(default_factory=list)
    #: 근거 블록이 차지한 글자 수. 제한과 비교해 보기 위한 값이다.
    evidence_chars: int = 0

    @property
    def injected_source_ids(self) -> list[str]:
        return self._source_ids(self.injected)

    @property
    def reference_source_ids(self) -> list[str]:
        return self._source_ids(self.reference)

    @staticmethod
    def _source_ids(cards: list[dict[str, Any]]) -> list[str]:
        seen: list[str] = []
        for card in cards:
            source = card.get("source_id")
            if source and source not in seen:
                seen.append(str(source))
        return seen


class ContextBuilder:
    def __init__(
        self,
        max_evidence_chars: int = 7000,
        max_recent_turns: int = 4,
        context_mode: str = "full",
        hybrid_full_cards: int = HYBRID_FULL_CARDS,
        ledger_min_bodies: int = 2,
    ):
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context_mode: {context_mode}")
        self.max_evidence_chars = max_evidence_chars
        self.max_recent_turns = max_recent_turns
        self.context_mode = context_mode
        self.hybrid_full_cards = hybrid_full_cards
        self.ledger_min_bodies = ledger_min_bodies

    def includes_body(self, index: int) -> bool:
        if self.context_mode == "full":
            return True
        if self.context_mode == "hybrid":
            return index < self.hybrid_full_cards
        return False

    def build(
        self,
        plan: QueryPlan,
        prefetch_cards: list[dict[str, Any]],
        recent_turns: list[dict[str, Any]],
        delivered: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """문자열만 필요할 때 쓴다. 기록이 필요하면 build_context()를 쓴다."""
        return self.build_context(plan, prefetch_cards, recent_turns, delivered).text

    def build_context(
        self,
        plan: QueryPlan,
        prefetch_cards: list[dict[str, Any]],
        recent_turns: list[dict[str, Any]],
        delivered: dict[str, dict[str, Any]] | None = None,
    ) -> BuiltContext:
        compact_turns = []
        for turn in recent_turns[-self.max_recent_turns :]:
            compact_turns.append(
                {
                    "turn_id": turn.get("turn_id", ""),
                    "user": str(turn.get("user_query", ""))[:300],
                    "answer_summary": str(turn.get("answer_summary", ""))[:500],
                    "query_intent": turn.get("query_intent", ""),
                    "project": turn.get("query_analysis", {}).get("filters", {}).get("project", ""),
                    "source_ids": turn.get("source_ids", []),
                    "tool_calls": [
                        {
                            "tool": call.get("tool", ""),
                            "tier": call.get("tier", ""),
                            "query": call.get("query", ""),
                            "result_count": call.get("result_count", 0),
                        }
                        for call in turn.get("tool_calls", [])[:3]
                    ],
                }
            )

        ledger = delivered or {}
        # 새로 볼 문서가 몇 장인지 먼저 센다. 부족하면 상위 참조를 본문으로
        # 되돌려, 한 턴에 본문 0장이 되는 일을 막는다.
        fresh_available = sum(
            1
            for card in prefetch_cards
            if str(card.get("source_ref", {}).get("document_id", "")) not in ledger
        )
        forced_bodies = (
            max(0, self.ledger_min_bodies - fresh_available) if ledger else 0
        )
        compact_cards: list[dict[str, Any]] = []
        injected_cards: list[dict[str, Any]] = []
        reference_cards: list[dict[str, Any]] = []
        dropped: list[str] = []
        used_chars = 0
        for index, card in enumerate(prefetch_cards):
            source_id = card.get("source_ref", {}).get("document_id")
            compact = {
                "evidence_id": card.get("evidence_id"),
                "tier": card.get("tier"),
                "title": card.get("title"),
                "date": card.get("date"),
                "project": card.get("project"),
                "summary": card.get("summary"),
                "source_id": source_id,
                "final_score": card.get("final_score"),
            }
            prior = ledger.get(str(source_id)) if source_id else None
            if prior is not None and forced_bodies > 0:
                # 새 문서가 부족한 턴이다. 순위가 높은 쪽부터 본문을 다시 준다.
                forced_bodies -= 1
                prior = None
            if prior is not None:
                # 앞선 턴에서 본문을 이미 전달한 문서다. 제목과 출처만 남긴다.
                # 목록에서 아예 빼지는 않는다 — 빼면 모델이 "그 문서는 없다"로
                # 읽고 같은 검색을 다시 한다(09-29 q_180에서 본 실패).
                compact["body_available"] = True
                compact["delivered_in_turn"] = prior.get("turn_index")
                compact["note"] = "이전 턴에 본문을 전달한 문서"
            elif self.includes_body(index):
                compact["content_excerpt"] = card.get("content_excerpt")
                # 같은 문서의 다른 대목까지 걸렸으면 함께 넣는다.
                # LTM 카드는 문서 단위라 대표 청크 하나만 보이는데, 문서 하나가
                # 청크 16~68개라 정답이 대표 청크 밖에 있는 경우가 실제로 있었다.
                extra = card.get("additional_excerpts") or []
                if extra:
                    compact["additional_excerpts"] = extra
            else:
                compact["body_available"] = True
            encoded = json.dumps(compact, ensure_ascii=False)
            if used_chars + len(encoded) > self.max_evidence_chars:
                # 남은 카드는 전부 빠진다. 순서가 곧 순위라 여기서 끊는 게 맞다.
                dropped.extend(
                    str(rest.get("evidence_id")) for rest in prefetch_cards[index:]
                )
                break
            compact_cards.append(compact)
            if prior is not None:
                reference_cards.append(compact)
            else:
                injected_cards.append(compact)
            used_chars += len(encoded)

        payload = {
            "query_plan": plan.to_dict(),
            "context_mode": self.context_mode,
            "recent_session_turns": compact_turns,
            "prefetched_evidence": compact_cards,
        }
        guidance = self._guidance(plan)
        if reference_cards:
            guidance += "\n" + (
                "delivered_in_turn이 붙은 근거는 앞선 턴에서 본문을 이미 전달한 "
                "문서다. 그 내용을 바탕으로 답하고, 본문을 다시 확인해야 하면 "
                "expand_evidence에 그 evidence_id를 넘긴다. "
                "같은 문서를 retrieve_memory로 다시 검색하지 않는다."
            )
        text = (
            "[RUNTIME_CONTEXT]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n[/RUNTIME_CONTEXT]\n"
            + guidance
        )
        return BuiltContext(
            text=text,
            injected=injected_cards,
            reference=reference_cards,
            dropped_evidence_ids=dropped,
            evidence_chars=used_chars,
        )

    def _guidance(self, plan: QueryPlan) -> str:
        if plan.answer_source == "session_only":
            return (
                "이번 질문은 최근 세션 맥락으로 답한다. prefetched_evidence가 비어 있으면 "
                "recent_session_turns의 user와 answer_summary를 기준으로 요약하고, "
                "조직 문서 근거가 필요해지는 경우에만 retrieve_memory를 호출한다."
            )
        if self.context_mode == "full":
            return (
                "위 컨텍스트는 이번 호출을 위해 선별된 자료다. 충분하면 바로 답하고, "
                "부족하거나 더 구체적인 근거가 필요하면 retrieve_memory를 호출한다."
            )
        return (
            "위 근거 중 body_available이 true인 항목은 요약만 제공했다. "
            "요약으로 판단할 수 있으면 그대로 답한다. "
            "원문 확인이 반드시 필요한 근거에 대해서만 expand_evidence를 호출하고, "
            "꼭 필요한 evidence_id만 지정한다. "
            "다른 주제의 근거가 더 필요하면 retrieve_memory를 호출한다."
        )
