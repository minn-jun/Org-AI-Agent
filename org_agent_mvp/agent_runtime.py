from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .config import AppConfig
from .context_builder import ContextBuilder
from .memory_store import MemoryStore
from .prefetch import MemoryPrefetcher
from .prompts import SYSTEM_PROMPT
from .query_analyzer import QueryAnalyzer, RuleBasedQueryAnalyzer
from .schemas import RETRIEVE_MEMORY_TOOL
from .session_store import SessionStore


class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass
class AgentTrace:
    llm_calls: int = 0
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_sources: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    session_id: str = ""
    query_analysis: dict[str, Any] = field(default_factory=dict)
    prefetch: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnLogger:
    log_dir: Path
    turn_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
    events: list[dict[str, Any]] = field(default_factory=list)
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append(
            {
                "index": len(self.events) + 1,
                "event": event,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "payload": payload,
            }
        )

    def write(self, data: dict[str, Any]) -> Path:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{self.turn_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def record_reasoning_step(self, payload: dict[str, Any]) -> None:
        self.reasoning_steps.append(
            {
                "step": len(self.reasoning_steps) + 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **payload,
            }
        )

    def record_tool_execution(self, payload: dict[str, Any]) -> None:
        self.tool_executions.append(
            {
                "step": len(self.tool_executions) + 1,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                **payload,
            }
        )


class AgentRuntime:
    def __init__(
        self,
        config: AppConfig,
        client: ChatClient,
        memory_store: MemoryStore,
        query_analyzer: QueryAnalyzer | None = None,
    ):
        self.config = config
        self.client = client
        self.memory_store = memory_store
        self.tools = [RETRIEVE_MEMORY_TOOL]
        self.query_analyzer = query_analyzer or RuleBasedQueryAnalyzer()
        self.prefetcher = MemoryPrefetcher(memory_store, total_top_k=config.prefetch_top_k)
        self.context_builder = ContextBuilder()
        self.session_store = SessionStore(
            config.project_root / "logs" / "sessions",
            cache_turns=config.session_cache_turns,
        )

    def run(
        self,
        user_query: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        log_dir: Path | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        session = self.session_store.load_or_create(session_id)
        recent_turns = list(session.get("turns", []))
        plan = self.query_analyzer.analyze(user_query, recent_turns)
        plan_payload = {
            **plan.to_dict(),
            "analyzer": {
                "type": type(self.query_analyzer).__name__,
                "model": getattr(self.query_analyzer, "model", ""),
            },
        }
        prefetch = self.prefetcher.prefetch(plan)
        runtime_context = self.context_builder.build(plan, prefetch.cards, recent_turns)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": runtime_context},
            {"role": "user", "content": user_query},
        ]
        trace = AgentTrace(
            session_id=session["session_id"],
            query_analysis=plan_payload,
            prefetch={
                "allocations": prefetch.allocations,
                "tier_result_counts": prefetch.tier_result_counts,
                "collected_count": prefetch.collected_count,
                "deduped_count": prefetch.deduped_count,
                "result_count": len(prefetch.cards),
                "sources": [
                    card.get("source_ref", {}).get("document_id", "")
                    for card in prefetch.cards
                ],
            },
        )
        turn_logger = TurnLogger(log_dir) if log_dir else None
        seen_tier_queries: set[tuple[str, str]] = set()

        self._emit(
            event_callback,
            turn_logger,
            "turn_start",
            {
                "query": user_query,
                "session_id": session["session_id"],
                "previous_turn_count": len(recent_turns),
            },
        )
        self._emit(event_callback, turn_logger, "query_analysis", plan_payload)
        self._emit(
            event_callback,
            turn_logger,
            "prefetch_start",
            {"allocations": prefetch.allocations, "query": plan.query_rewrites[-1]},
        )
        self._emit(
            event_callback,
            turn_logger,
            "prefetch_end",
            {
                "result_count": len(prefetch.cards),
                "tier_result_counts": prefetch.tier_result_counts,
                "collected_count": prefetch.collected_count,
                "deduped_count": prefetch.deduped_count,
                "sources": [
                    {
                        "tier": card.get("tier"),
                        "document_id": card.get("source_ref", {}).get("document_id", ""),
                        "score": card.get("rerank_score"),
                    }
                    for card in prefetch.cards
                ],
            },
        )
        self._emit(
            event_callback,
            turn_logger,
            "context_built",
            {
                "recent_turn_count": min(len(recent_turns), 4),
                "evidence_count": len(prefetch.cards),
                "context_chars": len(runtime_context),
            },
        )
        for card in prefetch.cards:
            source = card.get("source_ref", {}).get("document_id")
            if source and source not in trace.final_sources:
                trace.final_sources.append(source)
        for _ in range(self.config.max_tool_calls + 1):
            trace.llm_calls += 1
            message_roles = [message.get("role", "") for message in messages]
            self._emit(
                event_callback,
                turn_logger,
                "llm_call_start",
                {
                    "llm_call": trace.llm_calls,
                    "message_count": len(messages),
                    "tool_count": len(self.tools),
                    "message_roles": message_roles,
                },
            )
            assistant_message = self.client.chat(
                messages,
                self.tools,
                model=self.config.agent_model,
                temperature=0.2,
            )
            tool_calls = assistant_message.get("tool_calls") or []
            decision_payload = self._build_decision_payload(
                llm_call=trace.llm_calls,
                decision="tool_call" if tool_calls else "final_answer",
                message_roles=message_roles,
                assistant_message=assistant_message,
                source_count=len(trace.final_sources),
            )
            if turn_logger:
                turn_logger.record_reasoning_step(decision_payload)
            trace.reasoning_steps.append(
                {
                    "step": len(trace.reasoning_steps) + 1,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **decision_payload,
                }
            )
            self._emit(event_callback, turn_logger, "llm_decision", decision_payload)
            self._emit(
                event_callback,
                turn_logger,
                "llm_call_end",
                {
                    "llm_call": trace.llm_calls,
                    "has_tool_calls": bool(tool_calls),
                    "tool_call_count": len(tool_calls),
                    "content_preview": (assistant_message.get("content") or "")[:160],
                    "assistant_message": assistant_message,
                },
            )

            if not tool_calls:
                content = assistant_message.get("content") or ""
                trace.stopped_reason = "final_answer"
                self._emit(
                    event_callback,
                    turn_logger,
                    "final_answer",
                    {
                        "stopped_reason": trace.stopped_reason,
                        "source_count": len(trace.final_sources),
                    },
                )
                result = {
                    "answer": content,
                    "trace": self._trace_dict(trace),
                    "messages": messages + [assistant_message],
                }
                self._save_session_turn(session, turn_logger, user_query, result)
                self._write_turn_log(turn_logger, user_query, result)
                return result

            messages.append(assistant_message)
            for tool_call in tool_calls:
                result_message = self._execute_tool_call(
                    tool_call, seen_tier_queries, trace, event_callback, turn_logger
                )
                messages.append(result_message)

            if len(trace.tool_calls) >= self.config.max_tool_calls:
                trace.stopped_reason = "max_tool_calls"
                self._emit(
                    event_callback,
                    turn_logger,
                    "tool_limit_reached",
                    {"max_tool_calls": self.config.max_tool_calls},
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool call limit reached. Use the collected evidence to answer. "
                            "If evidence is insufficient, say so clearly."
                        ),
                    }
                )

        trace.stopped_reason = "loop_exhausted"
        self._emit(event_callback, turn_logger, "loop_exhausted", {})
        result = {
            "answer": "도구 호출 제한에 도달했지만 최종 답변을 생성하지 못했습니다.",
            "trace": self._trace_dict(trace),
            "messages": messages,
        }
        self._save_session_turn(session, turn_logger, user_query, result)
        self._write_turn_log(turn_logger, user_query, result)
        return result

    def _save_session_turn(
        self,
        session: dict[str, Any],
        turn_logger: TurnLogger | None,
        user_query: str,
        result: dict[str, Any],
    ) -> None:
        turn_id = turn_logger.turn_id if turn_logger else (
            datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
        )
        session_path = self.session_store.append_turn(
            session,
            {
                "turn_id": turn_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "user_query": user_query,
                "answer_summary": str(result.get("answer", ""))[:700],
                "source_ids": result["trace"].get("final_sources", []),
                "query_intent": result["trace"].get("query_analysis", {}).get("intent", ""),
                "query_analysis": result["trace"].get("query_analysis", {}),
                "prefetch": result["trace"].get("prefetch", {}),
                "reasoning_steps": result["trace"].get("reasoning_steps", []),
                "tool_calls": result["trace"].get("tool_calls", []),
                "stopped_reason": result["trace"].get("stopped_reason", ""),
            },
        )
        result["session_id"] = session["session_id"]
        result["session_log_path"] = str(session_path)

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        seen_tier_queries: set[tuple[str, str]],
        trace: AgentTrace,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        turn_logger: TurnLogger | None = None,
    ) -> dict[str, Any]:
        function = tool_call.get("function", {})
        name = function.get("name")
        tool_call_id = tool_call.get("id", "unknown_tool_call")
        if name != "retrieve_memory":
            content = {"error": f"Unsupported tool: {name}"}
            if turn_logger:
                turn_logger.record_tool_execution(
                    {
                        "tool": str(name),
                        "tool_call_id": tool_call_id,
                        "status": "unsupported_tool",
                        "error": content["error"],
                    }
                )
            return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(content)}

        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            content = {"error": f"Invalid JSON arguments: {exc}"}
            if turn_logger:
                turn_logger.record_tool_execution(
                    {
                        "tool": str(name),
                        "tool_call_id": tool_call_id,
                        "status": "invalid_arguments",
                        "error": content["error"],
                    }
                )
            return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(content)}

        tier = str(args.get("tier", "all")).lower()
        query = str(args.get("query", "")).strip()
        self._emit(
            event_callback,
            turn_logger,
            "tool_call_start",
            {
                "tool": name,
                "tool_call_id": tool_call_id,
                "raw_tool_call": tool_call,
                "arguments": args,
                "tier": tier,
                "query": query,
                "filters": args.get("filters") or {},
                "top_k": int(args.get("top_k") or self.config.default_top_k),
                "reason": args.get("reason", ""),
            },
        )
        repeat_key = (tier, query.lower())
        if repeat_key in seen_tier_queries:
            result = {
                "query": query,
                "tier": tier,
                "result_count": 0,
                "results": [],
                "warning": "Repeated tier/query blocked.",
            }
        else:
            seen_tier_queries.add(repeat_key)
            result = self.memory_store.retrieve(
                tier=tier,
                query=query,
                filters=args.get("filters") or {},
                top_k=int(args.get("top_k") or self.config.default_top_k),
            )
        self._emit(
            event_callback,
            turn_logger,
            "tool_call_end",
            {
                "tool": name,
                "tool_call_id": tool_call_id,
                "tier": tier,
                "query": query,
                "result_count": result.get("result_count", 0),
                "result": result,
                "sources": [
                    card.get("source_ref", {}).get("document_id", "")
                    for card in result.get("results", [])
                ],
                "warning": result.get("warning", ""),
            },
        )
        sources = [
            card.get("source_ref", {}).get("document_id", "")
            for card in result.get("results", [])
        ]
        if turn_logger:
            turn_logger.record_tool_execution(
                {
                    "tool": "retrieve_memory",
                    "tool_call_id": tool_call_id,
                    "status": "completed",
                    "tier": tier,
                    "query": query,
                    "filters": args.get("filters") or {},
                    "top_k": int(args.get("top_k") or self.config.default_top_k),
                    "reason": args.get("reason", ""),
                    "result_count": result.get("result_count", 0),
                    "sources": sources,
                    "warning": result.get("warning", ""),
                }
            )

        trace.tool_calls.append(
            {
                "tool": "retrieve_memory",
                "tier": tier,
                "query": query,
                "reason": args.get("reason", ""),
                "result_count": result.get("result_count", 0),
            }
        )
        for card in result.get("results", []):
            source = card.get("source_ref", {}).get("document_id")
            if source and source not in trace.final_sources:
                trace.final_sources.append(source)

        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "retrieve_memory",
            "content": json.dumps(result, ensure_ascii=False),
        }

    def _trace_dict(self, trace: AgentTrace) -> dict[str, Any]:
        return {
            "session_id": trace.session_id,
            "query_analysis": trace.query_analysis,
            "prefetch": trace.prefetch,
            "llm_calls": trace.llm_calls,
            "reasoning_steps": trace.reasoning_steps,
            "tool_calls": trace.tool_calls,
            "final_sources": trace.final_sources,
            "stopped_reason": trace.stopped_reason,
        }

    def _build_decision_payload(
        self,
        llm_call: int,
        decision: str,
        message_roles: list[str],
        assistant_message: dict[str, Any],
        source_count: int,
    ) -> dict[str, Any]:
        tool_calls = assistant_message.get("tool_calls") or []
        if tool_calls:
            summarized_calls = [self._summarize_tool_call(tool_call) for tool_call in tool_calls]
            reasons = [
                str(call.get("reason", "")).strip()
                for call in summarized_calls
                if str(call.get("reason", "")).strip()
            ]
            return {
                "llm_call": llm_call,
                "decision": decision,
                "reasoning_summary": " / ".join(reasons)
                or "모델이 현재 컨텍스트만으로는 답변 근거가 부족하다고 판단해 메모리 검색을 선택했다.",
                "next_action": "execute_tool",
                "message_roles": message_roles,
                "tool_calls": summarized_calls,
            }

        content = assistant_message.get("content") or ""
        return {
            "llm_call": llm_call,
            "decision": decision,
            "reasoning_summary": "모델이 추가 tool_call 없이 지금까지 모인 컨텍스트와 근거로 최종 답변이 가능하다고 판단했다.",
            "next_action": "return_answer",
            "message_roles": message_roles,
            "source_count_before_answer": source_count,
            "content_preview": str(content)[:220],
        }

    def _summarize_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        function = tool_call.get("function", {})
        raw_args = function.get("arguments") or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw_arguments": raw_args}
        return {
            "tool_call_id": tool_call.get("id", "unknown_tool_call"),
            "tool": function.get("name", ""),
            "tier": args.get("tier", ""),
            "query": args.get("query", ""),
            "filters": args.get("filters") or {},
            "top_k": args.get("top_k", self.config.default_top_k),
            "reason": args.get("reason", ""),
        }

    def _emit(
        self,
        event_callback: Callable[[str, dict[str, Any]], None] | None,
        turn_logger: TurnLogger | None,
        event: str,
        payload: dict[str, Any],
    ) -> None:
        if turn_logger:
            turn_logger.record(event, payload)
        if event_callback:
            event_callback(event, payload)

    def _write_turn_log(
        self,
        turn_logger: TurnLogger | None,
        user_query: str,
        result: dict[str, Any],
    ) -> None:
        if not turn_logger:
            return
        path = turn_logger.write(
            {
                "turn_id": turn_logger.turn_id,
                "user_query": user_query,
                "summary": {
                    "stopped_reason": result["trace"]["stopped_reason"],
                    "llm_calls": result["trace"]["llm_calls"],
                    "tool_call_count": len(result["trace"]["tool_calls"]),
                    "source_count": len(result["trace"]["final_sources"]),
                },
                "reasoning_steps": turn_logger.reasoning_steps,
                "tool_executions": turn_logger.tool_executions,
                "trace": result["trace"],
                "answer": result["answer"],
                "transcript": result["messages"],
                "debug_events": turn_logger.events,
            }
        )
        result["turn_log_path"] = str(path)
