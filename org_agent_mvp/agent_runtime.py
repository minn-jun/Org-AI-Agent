from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4

from .config import AppConfig
from .context_builder import ContextBuilder
from .memory_store import MAX_RETRIEVE_TOP_K, MemoryStore
from .prefetch import MemoryPrefetcher, prefetch_query
from .prompts import system_prompt
from .query_analyzer import QueryAnalyzer, RuleBasedQueryAnalyzer
from .schemas import EXPAND_EVIDENCE_TOOL, RETRIEVE_MEMORY_TOOL
from .session_store import SessionStore


class ChatClient(Protocol):
    """chat()은 {"message": ..., "usage": ...} 형태를 돌려준다.

    usage에는 prompt_tokens, completion_tokens, total_tokens, estimated가 들어간다.
    mock 클라이언트는 문자 수 기반 근사값을 쓰므로 estimated=True로 표시한다.
    """

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


def _empty_usage() -> dict[str, Any]:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@dataclass
class TokenUsage:
    """한 턴에서 쓴 토큰을 analyzer와 agent로 나눠 집계한다."""

    analyzer: dict[str, Any] = field(default_factory=_empty_usage)
    agent: dict[str, Any] = field(default_factory=_empty_usage)
    analyzer_calls: int = 0
    agent_calls: int = 0
    estimated: bool = False

    def add(self, bucket: str, usage: dict[str, Any] | None) -> None:
        usage = usage or {}
        target = getattr(self, bucket)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            target[key] += int(usage.get(key) or 0)
        if usage.get("estimated"):
            self.estimated = True

    def to_dict(self) -> dict[str, Any]:
        total = {
            key: self.analyzer[key] + self.agent[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        return {
            **total,
            # analyzer 호출까지 포함한 턴 전체 LLM 호출 수
            "llm_calls": self.analyzer_calls + self.agent_calls,
            "analyzer_calls": self.analyzer_calls,
            "agent_calls": self.agent_calls,
            "analyzer": dict(self.analyzer),
            "agent": dict(self.agent),
            "estimated": self.estimated,
        }


@dataclass
class AgentTrace:
    """한 턴의 실행 기록.

    근거를 세 단계로 나눠 적는다. 예전에는 한 칸(`final_sources`)에 섞여 있어서
    "찾았다"와 "전달했다"와 "썼다"를 구분할 수 없었다.

      retrieved_sources  검색이 찾아낸 후보 전체
      injected_sources   1차 컨텍스트에 **실제로 들어간** 근거
      tool_sources       실행 중 retrieve_memory로 추가된 근거
      final_sources      모델이 받은 전체 (injected + tool)
      cited_sources      답변 본문에 제목이나 출처가 나타난 근거 (문자열 일치 추정)

    2026-09-28 Allganize 20문항에서 후보 8건 중 6~7건만 주입된 문항이 15건이었다.
    그때 기록은 8건 전부를 근거로 적고 있었다.
    """

    llm_calls: int = 0
    reasoning_steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    injected_sources: list[str] = field(default_factory=list)
    tool_sources: list[str] = field(default_factory=list)
    dropped_evidence_ids: list[str] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    #: 상한에 걸려 실행하지 않은 도구 요청 수.
    skipped_tool_calls: int = 0
    #: 앞선 턴에 본문을 전달한 문서(세션 전달 원장). 원장이 꺼져 있으면 빈 목록이다.
    session_delivered_sources: list[str] = field(default_factory=list)
    #: 그중 이번 턴 컨텍스트에 **참조 카드로만** 들어간 문서.
    reference_sources: list[str] = field(default_factory=list)
    #: 이번 턴에 본문을 전달한 카드 원본. 턴이 끝날 때 원장에 적는다.
    delivered_cards: list[dict[str, Any]] = field(default_factory=list)
    #: source_id -> 제목. 인용 판정에만 쓴다.
    source_titles: dict[str, str] = field(default_factory=dict)
    final_sources: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    session_id: str = ""
    query_analysis: dict[str, Any] = field(default_factory=dict)
    prefetch: dict[str, Any] = field(default_factory=dict)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    expanded_ids: list[str] = field(default_factory=list)
    context_mode: str = "full"


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
        if config.context_mode != "full" or config.session_evidence_ledger:
            # full 모드는 원문을 이미 전부 넣으므로 확장할 것이 없다.
            # 단 전달 원장을 켜면 이전 턴 문서가 본문 없이 들어오므로,
            # full 모드에서도 되불러오는 도구가 있어야 한다. 없으면 참조만
            # 보여 주고 본문 경로를 막는 셈이 된다.
            self.tools.append(EXPAND_EVIDENCE_TOOL)
        self.query_analyzer = query_analyzer or RuleBasedQueryAnalyzer()
        self.prefetcher = MemoryPrefetcher(
            memory_store,
            total_top_k=config.prefetch_top_k,
            alpha=config.tier_prior_alpha,
            pool_per_tier=config.prefetch_pool_per_tier,
            cut_ratio=config.prefetch_cut_ratio,
            min_cards=config.prefetch_min_cards,
            tier_floor=config.prefetch_tier_floor,
        )
        self.context_builder = ContextBuilder(
            context_mode=config.context_mode,
            max_evidence_chars=config.max_evidence_chars,
            ledger_min_bodies=config.session_ledger_min_bodies,
        )
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
        # 저장본은 전체를 유지하고, 컨텍스트에는 최근 구간만 넘긴다.
        recent_turns = self.session_store.recent(session)
        plan = self.query_analyzer.analyze(user_query, recent_turns)
        plan_payload = {
            **plan.to_dict(),
            "analyzer": {
                "type": type(self.query_analyzer).__name__,
                "model": getattr(self.query_analyzer, "model", ""),
                "fallback_used": bool(
                    getattr(self.query_analyzer, "last_fallback_used", False)
                ),
            },
        }
        # 앞선 턴에 본문을 전달한 문서. 원장이 꺼져 있으면 빈 dict이고,
        # 그때는 아래 prefetch와 build_context가 예전과 똑같이 동작한다.
        ledger = (
            self.session_store.delivered(session)
            if self.config.session_evidence_ledger
            else {}
        )
        # 상한을 새 문서 기준으로 세게 한다. 참조로 줄어든 자리를 새 근거가
        # 채우지 않으면, 원장은 정보를 줄이기만 하고 왕복을 늘린다.
        prefetch = self.prefetcher.prefetch(plan, delivered=set(ledger))
        built = self.context_builder.build_context(
            plan, prefetch.cards, recent_turns, delivered=ledger
        )
        runtime_context = built.text
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt()},
            {"role": "system", "content": runtime_context},
            {"role": "user", "content": user_query},
        ]
        trace = AgentTrace(
            session_id=session["session_id"],
            query_analysis=plan_payload,
            prefetch={
                "tier_priors": prefetch.tier_priors,
                "pool_per_tier": prefetch.pool_per_tier,
                "tier_result_counts": prefetch.tier_result_counts,
                "collected_count": prefetch.collected_count,
                "deduped_count": prefetch.deduped_count,
                "max_raw_score": prefetch.max_raw_score,
                "cut_threshold": prefetch.cut_threshold,
                "result_count": len(prefetch.cards),
                "sources": [
                    card.get("source_ref", {}).get("document_id", "")
                    for card in prefetch.cards
                ],
            },
        )
        analyzer_usage = getattr(self.query_analyzer, "last_usage", {}) or {}
        trace.tokens.add("analyzer", analyzer_usage)
        # 규칙 기반 analyzer는 LLM을 호출하지 않으므로 0으로 남는다.
        trace.tokens.analyzer_calls = 1 if analyzer_usage else 0
        trace.context_mode = self.config.context_mode
        trace.session_delivered_sources = list(ledger)
        # expand_evidence는 재검색이 아니라 이 색인을 조회한다.
        # 원장을 켰으면 **이전 턴에 전달한 근거도** 되불러올 수 있어야 한다.
        evidence_index = self.session_store.delivered_cards(session) if ledger else {}
        evidence_index.update(
            {str(card["evidence_id"]): card for card in prefetch.cards}
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
            {
                "tier_priors": prefetch.tier_priors,
                "pool_per_tier": prefetch.pool_per_tier,
                "query": prefetch_query(plan),
            },
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
                "max_raw_score": prefetch.max_raw_score,
                "cut_threshold": prefetch.cut_threshold,
                "sources": [
                    {
                        "tier": card.get("tier"),
                        "document_id": card.get("source_ref", {}).get("document_id", ""),
                        "raw_score": card.get("retrieval_score"),
                        "normalized_score": card.get("normalized_score"),
                        "tier_prior": card.get("tier_prior"),
                        "final_score": card.get("final_score"),
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
                # candidate와 injected를 나눠 적는다. 예전의 evidence_count는
                # candidate 값이었는데 이름만 보고 주입 건수로 읽히고 있었다.
                "candidate_count": len(prefetch.cards),
                "injected_count": len(built.injected),
                "dropped_count": len(built.dropped_evidence_ids),
                "dropped_evidence_ids": built.dropped_evidence_ids,
                "evidence_chars": built.evidence_chars,
                "max_evidence_chars": self.config.max_evidence_chars,
                "context_chars": len(runtime_context),
            },
        )
        for card in prefetch.cards:
            source = card.get("source_ref", {}).get("document_id")
            if not source:
                continue
            trace.source_titles.setdefault(str(source), str(card.get("title", "")))
            if source not in trace.retrieved_sources:
                trace.retrieved_sources.append(source)
        trace.injected_sources = built.injected_source_ids
        trace.reference_sources = built.reference_source_ids
        trace.dropped_evidence_ids = list(built.dropped_evidence_ids)
        # 모델이 받은 근거만 final_sources에 넣는다. 도구 결과는 실행할 때 더한다.
        trace.final_sources = list(trace.injected_sources)
        # 원장에 적을 것은 **본문을 전달한** 카드뿐이다. 참조만 준 문서는
        # 이미 원장에 들어 있고, 다시 적으면 처음 전달한 턴 번호가 지워진다.
        injected_ids = {str(card.get("evidence_id")) for card in built.injected}
        trace.delivered_cards = [
            card
            for card in prefetch.cards
            if str(card.get("evidence_id")) in injected_ids
        ]
        for _ in range(self.config.max_tool_calls + 1):
            trace.llm_calls += 1
            message_roles = [message.get("role", "") for message in messages]
            # 상한에 도달했으면 도구 목록을 빼고 호출한다.
            # 예전에는 안내 메시지만 덧붙이고 도구를 계속 넘겼다. 안내는 지킬
            # 의무가 없는 요청이라 실제 제한이 아니었다 — 상한 3회인데 8회가
            # 실행되는 것을 재현했다. 여기서는 호출 자체에서 뺀다.
            tools_left = self.config.max_tool_calls - len(trace.tool_calls)
            tools_for_call = self.tools if tools_left > 0 else None
            self._emit(
                event_callback,
                turn_logger,
                "llm_call_start",
                {
                    "llm_call": trace.llm_calls,
                    "message_count": len(messages),
                    "tool_count": len(tools_for_call or []),
                    "tool_calls_left": max(0, tools_left),
                    "message_roles": message_roles,
                },
            )
            chat_response = self.client.chat(
                messages,
                tools_for_call,
                model=self.config.agent_model,
                temperature=0.2,
            )
            assistant_message = chat_response.get("message") or {}
            call_usage = chat_response.get("usage") or {}
            trace.tokens.add("agent", call_usage)
            trace.tokens.agent_calls = trace.llm_calls
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
                    "usage": call_usage,
                },
            )

            if not tool_calls:
                content = assistant_message.get("content") or ""
                trace.stopped_reason = "final_answer"
                trace.cited_sources = self._cited_sources(content, trace)
                self._emit(
                    event_callback,
                    turn_logger,
                    "final_answer",
                    {
                        "stopped_reason": trace.stopped_reason,
                        "retrieved_count": len(trace.retrieved_sources),
                        "injected_count": len(trace.injected_sources),
                        "source_count": len(trace.final_sources),
                        "cited_count": len(trace.cited_sources),
                        "tokens": trace.tokens.to_dict(),
                        "context_mode": trace.context_mode,
                        "expansion_count": len(trace.expanded_ids),
                    },
                )
                result = {
                    "answer": content,
                    "trace": self._trace_dict(trace),
                    "messages": messages + [assistant_message],
                }
                self._save_session_turn(
                    session, turn_logger, user_query, result, trace
                )
                self._write_turn_log(turn_logger, user_query, result)
                return result

            messages.append(assistant_message)
            for tool_call in tool_calls:
                # 한 응답에 도구 요청이 여러 개 들어오면 예전에는 전부 실행했다.
                # 남은 횟수만 실행하고 나머지는 실행하지 않은 것으로 응답한다.
                # tool_call마다 응답 메시지는 반드시 붙여야 한다 — 빠지면
                # 다음 호출에서 대화 형식이 깨진다.
                if len(trace.tool_calls) >= self.config.max_tool_calls:
                    trace.skipped_tool_calls += 1
                    messages.append(self._tool_limit_message(tool_call, turn_logger))
                    continue
                result_message = self._execute_tool_call(
                    tool_call,
                    seen_tier_queries,
                    evidence_index,
                    trace,
                    event_callback,
                    turn_logger,
                )
                messages.append(result_message)

            if len(trace.tool_calls) >= self.config.max_tool_calls:
                trace.stopped_reason = "max_tool_calls"
                self._emit(
                    event_callback,
                    turn_logger,
                    "tool_limit_reached",
                    {
                        "max_tool_calls": self.config.max_tool_calls,
                        "executed": len(trace.tool_calls),
                        "skipped": trace.skipped_tool_calls,
                    },
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
        self._save_session_turn(session, turn_logger, user_query, result, trace)
        self._write_turn_log(turn_logger, user_query, result)
        return result

    def _save_session_turn(
        self,
        session: dict[str, Any],
        turn_logger: TurnLogger | None,
        user_query: str,
        result: dict[str, Any],
        trace: "AgentTrace | None" = None,
    ) -> None:
        turn_id = turn_logger.turn_id if turn_logger else (
            datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
        )
        # 전달 원장은 append_turn **앞**에서 넣는다. append_turn이 저장까지
        # 하므로 같은 쓰기에 함께 담긴다.
        if self.config.session_evidence_ledger and trace is not None:
            self.session_store.record_delivered(
                session,
                trace.delivered_cards,
                turn_index=len(session.get("turns", [])) + 1,
                turn_id=turn_id,
                max_docs=self.config.session_ledger_max_docs,
                excerpt_chars=self.config.ltm_excerpt_chars,
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

    def _execute_expand_evidence(
        self,
        tool_call: dict[str, Any],
        evidence_index: dict[str, dict[str, Any]],
        trace: AgentTrace,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        turn_logger: TurnLogger | None = None,
    ) -> dict[str, Any]:
        """이미 제시한 근거의 원문만 돌려준다. 새로 검색하지 않는다."""
        tool_call_id = tool_call.get("id", "unknown_tool_call")
        function = tool_call.get("function", {})
        try:
            args = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            content = {"error": f"Invalid JSON arguments: {exc}"}
            if turn_logger:
                turn_logger.record_tool_execution(
                    {
                        "tool": "expand_evidence",
                        "tool_call_id": tool_call_id,
                        "status": "invalid_arguments",
                        "error": content["error"],
                    }
                )
            return {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(content, ensure_ascii=False),
            }

        requested = [str(item) for item in (args.get("evidence_ids") or [])]
        reason = str(args.get("reason", ""))
        self._emit(
            event_callback,
            turn_logger,
            "expand_start",
            {
                "tool_call_id": tool_call_id,
                "evidence_ids": requested,
                "reason": reason,
                "available": len(evidence_index),
            },
        )

        expanded: list[dict[str, Any]] = []
        unknown: list[str] = []
        for evidence_id in requested:
            card = evidence_index.get(evidence_id)
            if card is None:
                unknown.append(evidence_id)
                continue
            ref = card.get("source_ref") or {}
            document_id = str(ref.get("document_id", ""))
            # 카드에 실린 발췌가 아니라 원문을 읽는다. 예전에는 카드의
            # content_excerpt를 그대로 돌려줘서, 원문을 요청해도 이미 본 것과
            # 같은 글자만 돌아왔다(둘 다 같은 600자였다).
            source = self.memory_store.chunk_context(
                document_id,
                int(ref.get("chunk_index") or 0),
                neighbors=1,
                max_chars=self.config.max_evidence_chars,
            )
            if source.get("found"):
                body = "\n\n".join(
                    chunk["content"] for chunk in source.get("chunks", [])
                )
                pages = sorted(
                    {
                        page
                        for chunk in source.get("chunks", [])
                        for page in chunk.get("page_nos", [])
                    }
                )
            else:
                # 원문을 못 찾으면 최소한 카드 발췌라도 준다. 빈 내용을 주면
                # 모델이 "근거 없음"으로 읽는다.
                body = str(card.get("content_excerpt", ""))
                pages = list(ref.get("page_nos") or [])
            expanded.append(
                {
                    "evidence_id": evidence_id,
                    "title": card.get("title"),
                    "date": card.get("date"),
                    "project": card.get("project"),
                    "source_id": document_id,
                    "page_nos": pages,
                    "content": body,
                    "content_chars": len(body),
                    "from_source": bool(source.get("found")),
                }
            )
            if evidence_id not in trace.expanded_ids:
                trace.expanded_ids.append(evidence_id)

        result = {
            "expanded": expanded,
            "expanded_count": len(expanded),
            "unknown_evidence_ids": unknown,
        }
        self._emit(
            event_callback,
            turn_logger,
            "expand_end",
            {
                "tool_call_id": tool_call_id,
                "expanded_count": len(expanded),
                "expanded_ids": [item["evidence_id"] for item in expanded],
                "unknown_evidence_ids": unknown,
            },
        )
        if turn_logger:
            turn_logger.record_tool_execution(
                {
                    "tool": "expand_evidence",
                    "tool_call_id": tool_call_id,
                    "status": "completed",
                    "evidence_ids": requested,
                    "expanded_count": len(expanded),
                    "unknown_evidence_ids": unknown,
                    "reason": reason,
                }
            )
        trace.tool_calls.append(
            {
                "tool": "expand_evidence",
                "evidence_ids": requested,
                "reason": reason,
                "result_count": len(expanded),
            }
        )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": "expand_evidence",
            "content": json.dumps(result, ensure_ascii=False),
        }

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        seen_tier_queries: set[tuple[str, str]],
        evidence_index: dict[str, dict[str, Any]],
        trace: AgentTrace,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
        turn_logger: TurnLogger | None = None,
    ) -> dict[str, Any]:
        function = tool_call.get("function", {})
        name = function.get("name")
        tool_call_id = tool_call.get("id", "unknown_tool_call")
        if name == "expand_evidence":
            return self._execute_expand_evidence(
                tool_call, evidence_index, trace, event_callback, turn_logger
            )
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
            requested_top_k = int(args.get("top_k") or self.config.default_top_k)
            top_k = requested_top_k
            if self.config.tool_result_top_k_cap > 0:
                top_k = min(top_k, self.config.tool_result_top_k_cap)
            # 중복을 뺄 거라면 **빼고 나서도 top_k장이 남도록** 넉넉히 받아 온다.
            # 안 그러면 중복만 지운 빈 목록이 나가고(실측 q_180은 5장 전부 중복),
            # 모델은 "찾아도 아무것도 없다"로 읽어 한 번 더 검색한다.
            fetch_k = top_k
            if self.config.tool_result_dedupe:
                fetch_k = min(MAX_RETRIEVE_TOP_K, max(top_k * 3, top_k + 5))
            result = self.memory_store.retrieve(
                tier=tier,
                query=query,
                filters=args.get("filters") or {},
                top_k=fetch_k,
            )
            # 자르는 것은 **여기**다. 검색 결과를 그대로 두고 모델에게 보낼 때만
            # 줄인다. 기록에 남는 result도 잘린 쪽이어야 한다 — 09-28에 정리한
            # "기록은 실제로 전달한 것"과 같은 규칙이다.
            result = self._trim_tool_result(result, trace, limit=top_k)
            result["requested_top_k"] = requested_top_k
            result["effective_top_k"] = top_k
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
                    # 모델에게 실제로 보낸 글자 수. 도구 예산이 먹었는지 여기서 본다.
                    "content_chars": sum(
                        len(str(card.get("content_excerpt") or ""))
                        + sum(
                            len(str(part.get("content") or ""))
                            for part in (card.get("additional_excerpts") or [])
                        )
                        for card in (result.get("results") or [])
                    ),
                    "tool_result_budget": {
                        "chunks_per_doc": self.config.tool_result_chunks_per_doc,
                        "excerpt_chars": self.config.tool_result_excerpt_chars,
                        "dedupe": self.config.tool_result_dedupe,
                        "top_k_cap": self.config.tool_result_top_k_cap,
                    },
                    "duplicates_removed": result.get("duplicates_removed", 0),
                    "requested_top_k": result.get("requested_top_k"),
                    "effective_top_k": result.get("effective_top_k"),
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
            if not source:
                continue
            trace.source_titles.setdefault(str(source), str(card.get("title", "")))
            if source not in trace.tool_sources:
                trace.tool_sources.append(source)
            if source not in trace.retrieved_sources:
                trace.retrieved_sources.append(source)
            if source not in trace.final_sources:
                trace.final_sources.append(source)
            # 도구가 가져온 카드도 본문을 전달한 것이므로 원장 대상이다.
            trace.delivered_cards.append(card)

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
            "tokens": trace.tokens.to_dict(),
            "context_mode": trace.context_mode,
            "expanded_ids": list(trace.expanded_ids),
            "expansion_count": len(trace.expanded_ids),
            "reasoning_steps": trace.reasoning_steps,
            "tool_calls": trace.tool_calls,
            "skipped_tool_calls": trace.skipped_tool_calls,
            # 근거 3분리. 자세한 뜻은 AgentTrace의 주석에 있다.
            "retrieved_sources": trace.retrieved_sources,
            "injected_sources": trace.injected_sources,
            "tool_sources": trace.tool_sources,
            "dropped_evidence_ids": trace.dropped_evidence_ids,
            "cited_sources": trace.cited_sources,
            # 턴 간 기록. 원장이 꺼져 있으면 둘 다 빈 목록이다.
            "session_delivered_sources": trace.session_delivered_sources,
            "reference_sources": trace.reference_sources,
            "final_sources": trace.final_sources,
            "stopped_reason": trace.stopped_reason,
        }

    def _trim_tool_result(
        self,
        result: dict[str, Any],
        trace: "AgentTrace | None" = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """도구 결과 카드를 도구용 예산으로 줄인 새 결과를 돌려준다.

        세 가지를 한다.

        1. **이미 전달한 문서 제거** (`tool_result_dedupe`)
        2. **장수 제한** (`limit`)
        3. **본문 길이·대목 수 제한** (`tool_result_*`)

        1차 컨텍스트와 도구 결과는 같은 검색 경로를 타므로 카드가 같은 깊이로
        만들어진다. 그런데 비용은 전혀 다르다 — 1차 컨텍스트는 카드 8장이고
        한 번 실리지만, 도구 결과는 10~20장이고 **호출마다 다시 실린다.**

        2026-09-29 실측: 1차 컨텍스트를 3배로 키우자 도구 결과도 2.4배가 되어
        입력 토큰의 46%를 차지했다. 그리고 그 카드 96장 중 66장(69%)이 이미
        전달한 문서였다. 정답에 기여한 근거는 1차 컨텍스트 쪽이었다.

        원본 카드는 건드리지 않는다. prefetch가 같은 객체를 쓸 수 있다.
        """
        chunks = max(1, self.config.tool_result_chunks_per_doc)
        chars = max(1, self.config.tool_result_excerpt_chars)
        cards = result.get("results") or []

        duplicates = 0
        if self.config.tool_result_dedupe and trace is not None:
            # 이미 전달한 것 = 1차 컨텍스트 + 앞선 도구 호출 결과.
            # (이 함수는 trace.tool_sources가 갱신되기 **전**에 불린다.)
            # 원장을 켰으면 **이전 턴에 전달한 문서**까지 중복으로 본다.
            delivered = (
                set(trace.session_delivered_sources)
                | set(trace.injected_sources)
                | set(trace.tool_sources)
            )
            kept: list[dict[str, Any]] = []
            for card in cards:
                source = str((card.get("source_ref") or {}).get("document_id", ""))
                if source and source in delivered:
                    duplicates += 1
                    continue
                delivered.add(source)
                kept.append(card)
            cards = kept
        if limit is not None and limit > 0:
            cards = cards[:limit]

        trimmed: list[dict[str, Any]] = []
        for card in cards:
            item = dict(card)
            body = item.get("content_excerpt")
            if isinstance(body, str) and len(body) > chars:
                item["content_excerpt"] = body[:chars]
            extra = [
                {**part, "content": str(part.get("content", ""))[:chars]}
                for part in (item.get("additional_excerpts") or [])[: chunks - 1]
            ]
            item["additional_excerpts"] = extra
            ref = item.get("source_ref")
            if isinstance(ref, dict) and "extra_chunk_indexes" in ref:
                ref = dict(ref)
                ref["extra_chunk_indexes"] = [
                    part.get("chunk_index") for part in extra
                ]
                item["source_ref"] = ref
            trimmed.append(item)
        return {
            **result,
            "results": trimmed,
            "result_count": len(trimmed),
            "duplicates_removed": duplicates,
        }

    def _tool_limit_message(
        self,
        tool_call: dict[str, Any],
        turn_logger: TurnLogger | None = None,
    ) -> dict[str, Any]:
        """상한 때문에 실행하지 않은 도구 요청에 붙이는 응답."""
        tool_call_id = tool_call.get("id", "unknown_tool_call")
        name = str(tool_call.get("function", {}).get("name", ""))
        content = {
            "error": "tool_call_limit_reached",
            "max_tool_calls": self.config.max_tool_calls,
            "message": "도구 호출 상한에 도달해 실행하지 않았다. 지금까지 모인 근거로 답한다.",
        }
        if turn_logger:
            turn_logger.record_tool_execution(
                {
                    "tool": name,
                    "tool_call_id": tool_call_id,
                    "status": "skipped_tool_limit",
                    "max_tool_calls": self.config.max_tool_calls,
                }
            )
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(content, ensure_ascii=False),
        }

    def _cited_sources(self, answer: str, trace: AgentTrace) -> list[str]:
        """답변 본문에 나타난 근거를 고른다.

        **추정이다.** 모델이 출처를 구조화해서 돌려주지 않으므로, 제목이나
        source_id가 답변 글자에 있는지로 판단한다. 제목이 짧으면 우연히
        맞을 수 있어서 8자 이상만 본다. 정확한 인용 기록은 모델이 출처를
        따로 돌려주게 만들어야 하고, 그건 프롬프트·스키마 변경이 필요하다.
        """
        if not answer:
            return []
        cited: list[str] = []
        for source in trace.final_sources:
            title = str(trace.source_titles.get(source, "")).strip()
            if source and str(source) in answer:
                cited.append(source)
            elif len(title) >= 8 and title in answer:
                cited.append(source)
        return cited

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
                    "tokens": result["trace"]["tokens"],
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
