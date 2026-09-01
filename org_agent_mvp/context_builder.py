from __future__ import annotations

import json
from typing import Any

from .query_analyzer import QueryPlan


#: 1차 컨텍스트에 원문을 얼마나 넣을지 정한다.
#: full   - 근거 원문을 전부 넣는다 (비교 기준선)
#: summary- 제목과 요약만 넣고 원문은 expand_evidence로 요청받는다
#: hybrid - 상위 몇 건만 원문을 넣고 나머지는 요약만 넣는다
CONTEXT_MODES = ("full", "summary", "hybrid")
HYBRID_FULL_CARDS = 2


class ContextBuilder:
    def __init__(
        self,
        max_evidence_chars: int = 7000,
        max_recent_turns: int = 4,
        context_mode: str = "full",
        hybrid_full_cards: int = HYBRID_FULL_CARDS,
    ):
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context_mode: {context_mode}")
        self.max_evidence_chars = max_evidence_chars
        self.max_recent_turns = max_recent_turns
        self.context_mode = context_mode
        self.hybrid_full_cards = hybrid_full_cards

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
    ) -> str:
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

        compact_cards: list[dict[str, Any]] = []
        used_chars = 0
        for index, card in enumerate(prefetch_cards):
            compact = {
                "evidence_id": card.get("evidence_id"),
                "tier": card.get("tier"),
                "title": card.get("title"),
                "date": card.get("date"),
                "project": card.get("project"),
                "summary": card.get("summary"),
                "source_id": card.get("source_ref", {}).get("document_id"),
                "final_score": card.get("final_score"),
            }
            if self.includes_body(index):
                compact["content_excerpt"] = card.get("content_excerpt")
            else:
                compact["body_available"] = True
            encoded = json.dumps(compact, ensure_ascii=False)
            if used_chars + len(encoded) > self.max_evidence_chars:
                break
            compact_cards.append(compact)
            used_chars += len(encoded)

        payload = {
            "query_plan": plan.to_dict(),
            "context_mode": self.context_mode,
            "recent_session_turns": compact_turns,
            "prefetched_evidence": compact_cards,
        }
        return (
            "[RUNTIME_CONTEXT]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n[/RUNTIME_CONTEXT]\n"
            + self._guidance(plan)
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
