from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent_runtime import AgentRuntime
from .config import AppConfig
from .memory_store import MemoryStore
from .mock_llm import MockLLMClient
from .openrouter_client import OpenRouterClient
from .query_analyzer import LLMQueryAnalyzer, RuleBasedQueryAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Organization knowledge agent MVP")
    parser.add_argument("--question", "-q", help="Single question to ask.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock LLM.")
    parser.add_argument("--trace", action="store_true", help="Print JSON trace.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print live ReAct events.")
    parser.add_argument("--save-log", action="store_true", help="Save full turn log as JSON.")
    parser.add_argument("--log-dir", type=Path, help="Directory for saved turn logs.")
    parser.add_argument("--session-id", help="Continue an existing saved session.")
    return parser


def build_runtime(config: AppConfig, use_mock: bool) -> AgentRuntime:
    client = MockLLMClient() if use_mock else OpenRouterClient(config)
    query_analyzer = (
        RuleBasedQueryAnalyzer()
        if use_mock
        else LLMQueryAnalyzer(client, model=config.query_analyzer_model)
    )
    memory_store = MemoryStore(config.memory_root)
    return AgentRuntime(
        config=config,
        client=client,
        memory_store=memory_store,
        query_analyzer=query_analyzer,
    )


def print_event(event: str, payload: dict) -> None:
    if event == "turn_start":
        print(f"\n[session] {payload['session_id']} (이전 턴 {payload['previous_turn_count']}개)")
        print(f"[turn] 사용자 질문: {payload['query']}")
    elif event == "query_analysis":
        print(
            f"[analyzer] intent={payload['intent']} "
            f"answer_source={payload.get('answer_source', '')} "
            f"memory_needed={payload['memory_needed']}"
        )
        analyzer = payload.get("analyzer") or {}
        if analyzer:
            label = analyzer.get("type", "")
            model = analyzer.get("model", "")
            print(f"           analyzer={label}" + (f" model={model}" if model else ""))
        print(f"           reason={payload['reason']}")
        weights = payload["memory_weights"]
        print(
            "           weights="
            f"STM {weights['stm']:.0%} / MTM {weights['mtm']:.0%} / LTM {weights['ltm']:.0%}"
        )
    elif event == "prefetch_start":
        allocation = payload["allocations"]
        if sum(allocation.values()) == 0:
            print("[prefetch] 메모리 검색 생략 (세션/직접 답변 흐름)")
            return
        print(
            "[prefetch] 후보 검색 시작 "
            f"(STM {allocation['stm']} / MTM {allocation['mtm']} / LTM {allocation['ltm']})"
        )
    elif event == "prefetch_end":
        if payload.get("result_count", 0) == 0 and payload.get("collected_count", 0) == 0:
            print("[prefetch] 컨텍스트 후보 0건")
            return
        counts = payload.get("tier_result_counts") or {}
        if counts:
            print(
                "[prefetch] 검색 결과 "
                f"(STM {counts.get('stm', 0)} / MTM {counts.get('mtm', 0)} / "
                f"LTM {counts.get('ltm', 0)})"
            )
        print(
            f"[prefetch] 수집 {payload.get('collected_count', payload['result_count'])}건 "
            f"→ 중복제거 {payload.get('deduped_count', payload['result_count'])}건 "
            f"→ 컨텍스트 후보 {payload['result_count']}건"
        )
        for source in payload.get("sources", [])[:8]:
            print(
                f"           [{source['tier']}] {source['document_id']} "
                f"score={source['score']}"
            )
    elif event == "context_built":
        print(
            f"[context] 최근 턴 {payload['recent_turn_count']}개 + "
            f"근거 {payload['evidence_count']}건 ({payload['context_chars']} chars)"
        )
    elif event == "llm_call_start":
        print(
            f"[llm #{payload['llm_call']}] 호출 시작 "
            f"(messages={payload['message_count']}, tools={payload['tool_count']})"
        )
    elif event == "llm_call_end":
        if payload["has_tool_calls"]:
            print(f"[llm #{payload['llm_call']}] tool_call {payload['tool_call_count']}개 요청")
        else:
            print(f"[llm #{payload['llm_call']}] 최종 답변 생성")
    elif event == "llm_decision":
        print(f"[reasoning #{payload['llm_call']}] decision={payload['decision']}")
    elif event == "tool_call_start":
        print(f"[tool] {payload['tool']} 시작")
        print(f"       tier={payload['tier']} top_k={payload['top_k']}")
        print(f"       query={payload['query']}")
        if payload.get("filters"):
            print(f"       filters={json.dumps(payload['filters'], ensure_ascii=False)}")
        if payload.get("reason"):
            print(f"       reason={payload['reason']}")
    elif event == "tool_call_end":
        print(f"[tool] 검색 완료: {payload['result_count']}건")
        for source in payload.get("sources", [])[:5]:
            print(f"       source={source}")
        if payload.get("warning"):
            print(f"       warning={payload['warning']}")
    elif event == "tool_limit_reached":
        print(f"[policy] tool call 제한 도달: {payload['max_tool_calls']}회")
    elif event == "final_answer":
        print(f"[final] 답변 준비 완료 (sources={payload['source_count']})")
    elif event == "loop_exhausted":
        print("[final] loop exhausted")


def ask_once(
    runtime: AgentRuntime,
    question: str,
    show_trace: bool,
    verbose: bool,
    log_dir: Path | None,
    session_id: str | None = None,
) -> dict | None:
    try:
        result = runtime.run(
            question,
            event_callback=print_event if verbose else None,
            log_dir=log_dir,
            session_id=session_id,
        )
    except RuntimeError as exc:
        print(f"\n[error]\n{exc}")
        return None
    print("\n[답변]\n")
    print(result["answer"])
    if result.get("turn_log_path"):
        print(f"\n[turn log]\n{result['turn_log_path']}")
    if show_trace:
        print("\n[trace]\n")
        print(json.dumps(result["trace"], ensure_ascii=False, indent=2))
    return result


def interactive(
    runtime: AgentRuntime,
    show_trace: bool,
    verbose: bool,
    log_dir: Path | None,
    session_id: str | None = None,
) -> None:
    active_session_id = session_id
    print("조직지식 에이전트 MVP입니다. exit/quit: 종료, /new: 새 세션")
    while True:
        question = input("\n질문> ").strip()
        if question.lower() in {"exit", "quit"}:
            return
        if question == "/new":
            active_session_id = None
            print("새 세션을 시작합니다.")
            continue
        if not question:
            continue
        result = ask_once(
            runtime, question, show_trace, verbose, log_dir, active_session_id
        )
        if result:
            active_session_id = result["session_id"]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig.load()
    try:
        runtime = build_runtime(config, use_mock=args.mock)
    except ValueError as exc:
        print(str(exc))
        print("OpenRouter 키를 .env에 넣거나 --mock 옵션으로 구조를 먼저 확인하세요.")
        return 2
    log_dir = None
    if args.save_log:
        log_dir = args.log_dir or (config.project_root / "logs" / "turns")

    if args.question:
        ask_once(
            runtime,
            args.question,
            args.trace,
            args.verbose,
            log_dir,
            args.session_id,
        )
    else:
        interactive(runtime, args.trace, args.verbose, log_dir, args.session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
