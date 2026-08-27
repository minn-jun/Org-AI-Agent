from __future__ import annotations

import json
from typing import Any

from .query_analyzer import QueryPlan


class ContextBuilder:
    def __init__(self, max_evidence_chars: int = 7000, max_recent_turns: int = 4):
        self.max_evidence_chars = max_evidence_chars
        self.max_recent_turns = max_recent_turns

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
        for card in prefetch_cards:
            compact = {
                "evidence_id": card.get("evidence_id"),
                "tier": card.get("tier"),
                "title": card.get("title"),
                "date": card.get("date"),
                "project": card.get("project"),
                "summary": card.get("summary"),
                "content_excerpt": card.get("content_excerpt"),
                "source_id": card.get("source_ref", {}).get("document_id"),
                "rerank_score": card.get("rerank_score"),
            }
            encoded = json.dumps(compact, ensure_ascii=False)
            if used_chars + len(encoded) > self.max_evidence_chars:
                break
            compact_cards.append(compact)
            used_chars += len(encoded)

        payload = {
            "query_plan": plan.to_dict(),
            "recent_session_turns": compact_turns,
            "prefetched_evidence": compact_cards,
        }
        if plan.answer_source == "session_only":
            guidance = (
                "이번 질문은 최근 세션 맥락으로 답한다. prefetched_evidence가 비어 있으면 "
                "recent_session_turns의 user와 answer_summary를 기준으로 요약하고, "
                "조직 문서 근거가 필요해지는 경우에만 retrieve_memory를 호출한다."
            )
        else:
            guidance = (
                "위 컨텍스트는 이번 호출을 위해 선별된 자료다. 충분하면 바로 답하고, "
                "부족하거나 더 구체적인 근거가 필요하면 retrieve_memory를 호출한다."
            )
        return (
            "[RUNTIME_CONTEXT]\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n[/RUNTIME_CONTEXT]\n"
            + guidance
        )
