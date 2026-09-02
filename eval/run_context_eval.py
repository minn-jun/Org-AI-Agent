"""컨텍스트 주입 방식(B) 비교.

같은 질문을 full / summary / hybrid 세 모드로 돌려
프롬프트 토큰, LLM 호출 수, 원문 확장률을 비교한다.

A방식 평가(run_eval.py)와 달리 실제 LLM 호출이 필요하다.
모델이 "원문을 요청할지"를 스스로 판단해야 측정이 성립하기 때문이다.

    python eval/run_context_eval.py --mock            # 흐름 확인 (무료, 토큰은 추정치)
    python eval/run_context_eval.py --limit 5         # 실모델, 앞 5건만
    python eval/run_context_eval.py --sleep 5         # rate limit 대비

mock의 토큰은 문자 수 기반 추정치이므로 절감률 비교에 쓰지 않는다.
결과의 estimated 표시를 반드시 확인한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from org_agent_mvp.agent_runtime import AgentRuntime  # noqa: E402
from org_agent_mvp.config import AppConfig  # noqa: E402
from org_agent_mvp.memory_store import MemoryStore  # noqa: E402
from org_agent_mvp.mock_llm import MockLLMClient  # noqa: E402
from org_agent_mvp.openrouter_client import OpenRouterClient  # noqa: E402
from org_agent_mvp.query_analyzer import (  # noqa: E402
    LLMQueryAnalyzer,
    RuleBasedQueryAnalyzer,
)

DEFAULT_CASES = PROJECT_ROOT / "tests" / "fixtures" / "eval_cases.jsonl"
MODES = ("full", "summary", "hybrid")


def build_runtime(config: AppConfig, use_mock: bool) -> AgentRuntime:
    client = MockLLMClient() if use_mock else OpenRouterClient(config)
    memory_store = MemoryStore(config.memory_root, filter_penalty=config.filter_penalty)
    analyzer = (
        RuleBasedQueryAnalyzer()
        if use_mock
        else LLMQueryAnalyzer(
            client,
            model=config.query_analyzer_model,
            max_tokens=config.query_analyzer_max_tokens,
            vocabulary=memory_store.filter_vocabulary(),
        )
    )
    return AgentRuntime(
        config=config,
        client=client,
        memory_store=memory_store,
        query_analyzer=analyzer,
    )


def seeded_session_id(runtime: AgentRuntime, recent_turns: list[dict[str, Any]]) -> str | None:
    """fixture의 recent_turns를 세션 캐시에 심어 후속 질문을 재현한다."""
    if not recent_turns:
        return None
    session = runtime.session_store.create()
    for index, turn in enumerate(recent_turns):
        runtime.session_store.append_turn(
            session,
            {
                "turn_id": f"seed-{index}",
                "timestamp": "2026-09-01T00:00:00",
                "user_query": turn.get("user_query", ""),
                "answer_summary": turn.get("answer_summary", ""),
                "source_ids": turn.get("source_ids", []),
                "query_intent": turn.get("query_intent", ""),
                "query_analysis": turn.get("query_analysis", {}),
                "prefetch": {},
                "reasoning_steps": [],
                "tool_calls": [],
                "stopped_reason": "final_answer",
            },
        )
    return str(session["session_id"])


def run_mode(
    cases: list[dict[str, Any]],
    mode: str,
    base_config: AppConfig,
    args: argparse.Namespace,
    workdir: Path,
) -> dict[str, Any]:
    config = replace(base_config, project_root=workdir / mode, context_mode=mode)
    runtime = build_runtime(config, args.mock)

    rows: list[dict[str, Any]] = []
    for case in cases:
        session_id = seeded_session_id(runtime, case.get("recent_turns") or [])
        try:
            result = runtime.run(case["query"], session_id=session_id)
        except RuntimeError as exc:
            rows.append({"id": case["id"], "error": str(exc)[:180]})
            if args.sleep:
                time.sleep(args.sleep)
            continue
        trace = result["trace"]
        tokens = trace["tokens"]
        rows.append(
            {
                "id": case["id"],
                "prompt_tokens": tokens["prompt_tokens"],
                "completion_tokens": tokens["completion_tokens"],
                "total_tokens": tokens["total_tokens"],
                "agent_calls": tokens["agent_calls"],
                "llm_calls": tokens["llm_calls"],
                "expansion_count": trace["expansion_count"],
                "expanded": bool(trace["expanded_ids"]),
                "evidence_count": trace["prefetch"]["result_count"],
                "analyzer_fallback": trace["query_analysis"]["analyzer"]["fallback_used"],
                "estimated": tokens["estimated"],
                "answer_chars": len(result["answer"]),
            }
        )
        if args.sleep:
            time.sleep(args.sleep)

    ok = [r for r in rows if "error" not in r]

    def mean(key: str) -> float:
        return round(sum(r[key] for r in ok) / len(ok), 1) if ok else 0.0

    return {
        "mode": mode,
        "cases": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "estimated": any(r.get("estimated") for r in ok),
        "prompt_tokens_mean": mean("prompt_tokens"),
        "prompt_tokens_sum": sum(r["prompt_tokens"] for r in ok),
        "total_tokens_mean": mean("total_tokens"),
        "agent_calls_mean": mean("agent_calls"),
        "llm_calls_mean": mean("llm_calls"),
        "expansion_rate": round(sum(1 for r in ok if r["expanded"]) / len(ok), 4) if ok else 0.0,
        "expansion_count_mean": mean("expansion_count"),
        "analyzer_fallbacks": sum(1 for r in ok if r["analyzer_fallback"]),
        "rows": rows,
    }


def print_report(results: list[dict[str, Any]]) -> None:
    base = next((r for r in results if r["mode"] == "full"), results[0])
    baseline = base["prompt_tokens_mean"] or 1.0
    estimated = any(r["estimated"] for r in results)

    print()
    print(
        f"{'모드':<9}{'프롬프트':>10}{'절감':>9}{'총토큰':>10}"
        f"{'agent호출':>10}{'확장률':>9}{'오류':>6}"
    )
    print("-" * 62)
    for r in results:
        delta = (r["prompt_tokens_mean"] - baseline) / baseline * 100
        print(
            f"{r['mode']:<9}{r['prompt_tokens_mean']:>10.0f}{delta:>8.1f}%"
            f"{r['total_tokens_mean']:>10.0f}{r['agent_calls_mean']:>10.1f}"
            f"{r['expansion_rate']:>8.0%}{r['errors']:>6}"
        )
    print("-" * 62)
    print("* 프롬프트/총토큰은 케이스당 평균, 절감은 full 대비")
    if estimated:
        print("* mock 추정치가 포함되어 있다. 절감률 비교에 쓰지 않는다.")
    fallbacks = sum(r["analyzer_fallbacks"] for r in results)
    if fallbacks:
        print(f"* analyzer가 규칙 기반으로 대체된 턴 {fallbacks}건. 해당 턴은 해석에 주의한다.")

    for r in results:
        if r["mode"] == "full" or not r["ok"]:
            continue
        expanded = [row["id"] for row in r["rows"] if row.get("expanded")]
        if expanded:
            print(f"\n[{r['mode']}] 원문을 요청한 질문 {len(expanded)}건: {', '.join(expanded)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="컨텍스트 주입 방식 비교")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--core",
        action="store_true",
        help="core로 태깅된 20건만 실행한다. 유료 측정에서 비용을 줄일 때 쓴다.",
    )
    parser.add_argument("--mock", action="store_true", help="LLM 없이 흐름만 확인")
    parser.add_argument("--limit", type=int, help="앞에서 N건만 실행")
    parser.add_argument("--sleep", type=float, default=0.0, help="호출 사이 대기 초")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--json", type=Path, help="결과를 JSON으로 저장")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.core:
        cases = [c for c in cases if c.get("core")]
    if args.limit:
        cases = cases[: args.limit]

    base_config = AppConfig.load()
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        workdir = Path(temp_dir)
        for mode in args.modes:
            print(f"실행 중: {mode} ({len(cases)}건)...")
            results.append(run_mode(cases, mode, base_config, args, workdir))

    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
