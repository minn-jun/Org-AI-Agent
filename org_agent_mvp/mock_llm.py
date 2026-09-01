from __future__ import annotations

import json
from typing import Any


class MockLLMClient:
    """Deterministic local model stub for testing the ReAct loop without an API key."""

    #: 한국어 혼합 텍스트 기준 대략치. 실제 토크나이저 값이 아니므로 비교 실험에는 쓰지 않는다.
    CHARS_PER_TOKEN = 2

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        message = self._respond(messages, tools, **kwargs)
        return {"message": message, "usage": self._estimate_usage(messages, message)}

    def _estimate_usage(
        self,
        messages: list[dict[str, Any]],
        message: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_chars = sum(len(str(item.get("content") or "")) for item in messages)
        completion_chars = len(str(message.get("content") or ""))
        for tool_call in message.get("tool_calls") or []:
            completion_chars += len(json.dumps(tool_call, ensure_ascii=False))
        prompt_tokens = prompt_chars // self.CHARS_PER_TOKEN
        completion_tokens = completion_chars // self.CHARS_PER_TOKEN
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        }

    def _respond(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        user_query = self._last_user_query(messages)
        tool_results = self._tool_results(messages)
        runtime_context = self._runtime_context(messages)
        prefetch_cards = runtime_context.get("prefetched_evidence", [])
        query_plan = runtime_context.get("query_plan", {})
        tried_tiers = {result.get("tier", "").lower() for result in tool_results}

        # B방식 경로. 요약만 받은 근거가 있으면 상위 2건의 원문을 요청한다.
        already_expanded = any("expanded" in result for result in tool_results)
        if already_expanded:
            return {
                "role": "assistant",
                "content": self._final_answer(user_query, tool_results, prefetch_cards),
            }
        if not tool_results:
            pending = [
                str(card.get("evidence_id"))
                for card in prefetch_cards
                if card.get("body_available") and card.get("evidence_id")
            ]
            if pending:
                return self._expand_call(
                    pending[:2],
                    f"요약만 제공된 근거 {len(pending)}건 중 상위 2건의 원문 확인",
                )

        if not tool_results:
            if query_plan.get("answer_source") == "session_only":
                return {
                    "role": "assistant",
                    "content": self._session_answer(runtime_context),
                }
            if not query_plan.get("memory_needed", True):
                return {
                    "role": "assistant",
                    "content": (
                        "이 질문은 조직 메모리 검색 없이 답변 가능한 일반 질문으로 분류되었습니다. "
                        "실제 OpenRouter 연결 시 모델이 현재 대화만으로 답변합니다."
                    ),
                }
            tier = self._first_tier(user_query)
            reason = (
                f"prefetch 후보 {len(prefetch_cards)}건을 확인했으며, "
                f"답변 근거를 구체화하기 위해 {tier.upper()}를 추가 탐색"
            )
            return self._tool_call(tier, user_query, reason, query_plan.get("filters", {}))

        if self._needs_comparison(user_query) and "ltm" not in tried_tiers:
            return self._tool_call(
                "ltm",
                user_query,
                "최근 결정과 공식 기준을 비교하기 위해 LTM 확인",
                query_plan.get("filters", {}),
            )

        if self._is_recent(user_query) and not self._has_results(tool_results) and "mtm" not in tried_tiers:
            return self._tool_call(
                "mtm",
                user_query,
                "STM에서 충분한 근거를 찾지 못해 MTM으로 확장",
                query_plan.get("filters", {}),
            )

        if not self._has_results(tool_results) and "ltm" not in tried_tiers:
            return self._tool_call(
                "ltm",
                user_query,
                "이전 tier에서 근거가 부족해 LTM으로 확장",
                query_plan.get("filters", {}),
            )

        return {
            "role": "assistant",
            "content": self._final_answer(user_query, tool_results, prefetch_cards),
        }

    def _last_user_query(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    def _tool_results(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                results.append(json.loads(str(message.get("content", "{}"))))
            except json.JSONDecodeError:
                pass
        return results

    def _runtime_context(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        for message in messages:
            content = str(message.get("content", ""))
            if "[RUNTIME_CONTEXT]" not in content:
                continue
            try:
                raw = content.split("[RUNTIME_CONTEXT]", 1)[1].split("[/RUNTIME_CONTEXT]", 1)[0]
                return json.loads(raw)
            except (IndexError, json.JSONDecodeError):
                continue
        return {}

    def _session_answer(self, runtime_context: dict[str, Any]) -> str:
        turns = runtime_context.get("recent_session_turns", [])
        if not turns:
            return "현재 세션에서 요약할 이전 대화를 찾지 못했습니다."

        lines = ["앞선 대화 요약입니다.", ""]
        for turn in turns[-3:]:
            user = str(turn.get("user", "")).strip()
            answer = str(turn.get("answer_summary", "")).strip()
            if user:
                lines.append(f"- 질문: {user}")
            if answer:
                lines.append(f"  답변 요지: {answer[:220]}")
        return "\n".join(lines)

    def _first_tier(self, query: str) -> str:
        if self._needs_comparison(query):
            return "stm"
        if any(word in query for word in ["아까", "방금", "오늘", "다음 일정", "action item"]):
            return "stm"
        if any(word in query for word in ["이번 달", "최근", "회의록", "초안", "보고서"]):
            return "mtm"
        if any(word in query for word in ["공식", "최종", "기준", "계획서", "회사"]):
            return "ltm"
        return "all"

    def _needs_comparison(self, query: str) -> bool:
        return any(word in query for word in ["충돌", "비교", "달라진", "차이"])

    def _is_recent(self, query: str) -> bool:
        return any(word in query for word in ["아까", "방금", "오늘", "최근"])

    def _has_results(self, tool_results: list[dict[str, Any]]) -> bool:
        return any(int(result.get("result_count", 0)) > 0 for result in tool_results)

    def _tool_call(
        self,
        tier: str,
        query: str,
        reason: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        arguments = {
            "tier": tier,
            "query": query,
            "filters": filters or {},
            "top_k": 5,
            "reason": reason,
        }
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"mock_call_{tier}",
                    "type": "function",
                    "function": {
                        "name": "retrieve_memory",
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }

    def _expand_call(self, evidence_ids: list[str], reason: str) -> dict[str, Any]:
        arguments = {"evidence_ids": evidence_ids, "reason": reason}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "mock_call_expand",
                    "type": "function",
                    "function": {
                        "name": "expand_evidence",
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        }

    def _final_answer(
        self,
        query: str,
        tool_results: list[dict[str, Any]],
        prefetch_cards: list[dict[str, Any]],
    ) -> str:
        cards: list[dict[str, Any]] = []
        for card in prefetch_cards:
            cards.append(
                {
                    **card,
                    "source_ref": {"document_id": card.get("source_id", "")},
                }
            )
        for result in tool_results:
            cards.extend(result.get("results", []))
        unique_cards: list[dict[str, Any]] = []
        seen: set[str] = set()
        for card in cards:
            source = card.get("source_ref", {}).get("document_id", "")
            if source in seen:
                continue
            seen.add(source)
            unique_cards.append(card)
        cards = unique_cards
        if not cards:
            return "확인 가능한 근거를 찾지 못했습니다. 현재 자료만으로는 해당 질문에 답하기 어렵습니다."

        lines = ["확인한 내용입니다.", ""]
        for card in cards[:4]:
            source = card.get("source_ref", {}).get("document_id", "")
            lines.append(
                f"- {card.get('title')}: {card.get('summary') or card.get('quote')} "
                f"(출처: {source})"
            )
        return "\n".join(lines)
