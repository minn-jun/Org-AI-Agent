"""계층 간 중복 제거가 **답변**까지 바꾸는지 본다 (LLM 호출).

    python scripts/llm_answer_check.py --mock            # 배선 확인 (비용 0)
    python scripts/llm_answer_check.py --count 8         # 실제 호출

## 왜 소량인가

299문항을 LLM으로 돌리면 비용이 크다. 중복 제거가 **근거 카드를 실제로 바꾼 질문**만 고른다
(`planted_nomerge` vs `planted` 결과 비교). 질문 하나당 설정 두 가지를 돌리므로 호출은 count × 2다.

## 무엇을 고정하나

질의 분석기는 **규칙 기반으로 고정**한다. LLM 분석기를 쓰면 실행마다 계획이 흔들려 답변 차이의 원인이 섞인다.
따라서 이 실행에서 LLM은 **답변 생성에만** 쓰인다.

결과: org_agent_mvp/logs/allganize/llm_answers.json (질문별 두 설정의 답변·근거 목록)
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent.parent
LOG_DIR = APP / "logs" / "allganize"
DATA = ROOT / "ltm"
SEED_DIR = ROOT / "seed_tiers_planted"

#: 비교 축. --axis로 고른다.
AXES: dict[str, dict[str, tuple[str, dict[str, str]]]] = {
    # 계층 간 중복 제거 전후
    "merge": {
        "nomerge": ("출처 병합 끔", {"PREFETCH_MERGE_SAME_SOURCE": "0"}),
        "merge": ("출처 병합 켬", {"PREFETCH_MERGE_SAME_SOURCE": "1"}),
    },
    # 근거를 프롬프트에 어떻게 넣는가. full은 8장 본문을 통째로, summary는 요약만 넣고
    # 모델이 필요할 때 expand_evidence 도구로 본문을 펼친다.
    "context": {
        "full": ("근거 전문 (full)", {"CONTEXT_MODE": "full"}),
        "summary": ("요약 + 필요시 확장 (summary)", {"CONTEXT_MODE": "summary"}),
        "hybrid": ("상위만 전문 (hybrid)", {"CONTEXT_MODE": "hybrid"}),
    },
    # prefetch를 "이미 끝난 검색"으로 알려 주면 재검색 왕복이 줄어드는가.
    # 2026-09-16 실측: prefetch 8장을 받고도 78%의 턴이 다시 검색했다.
    "prompt": {
        "hint_off": ("기본 프롬프트", {"PROMPT_PREFETCH_HINT": "0"}),
        "hint_on": ("prefetch 안내 추가", {"PROMPT_PREFETCH_HINT": "1"}),
    },
}
CONFIGS = AXES["merge"]


def _cases() -> dict[str, dict]:
    rows = {}
    for name in ("dev", "test"):
        with (DATA / f"cases_{name}.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    case = json.loads(line)
                    rows[case["qid"]] = case
    return rows


def _changed_qids(limit: int) -> list[str]:
    """중복 제거로 근거 카드가 달라진 질문. 병합으로 한 장이 된 질문을 먼저 고른다."""
    merged = json.loads((LOG_DIR / "planted_planted.json").read_text(encoding="utf-8"))["planted_rows"]
    plain = {r["qid"]: r for r in json.loads((LOG_DIR / "planted_planted_nomerge.json").read_text(encoding="utf-8"))["planted_rows"]}
    folded = [r["qid"] for r in merged if r.get("merged_rank")]
    moved = [r["qid"] for r in merged if not r.get("merged_rank") and plain.get(r["qid"], {}).get("first") != r["first"]]
    return (folded + moved)[:limit]


#: 네트워크가 끊길 때 올라오는 예외들. 2026-09-16 실제로 두 가지가 나왔다.
#:   urllib.error.URLError: SSL EOF        -> openrouter_client가 RuntimeError로 감싼다
#:   http.client.RemoteDisconnected        -> **감싸지 않아 그대로 올라온다**
#: 그래서 RuntimeError만 잡으면 재시도가 동작하지 않는다.
RETRY_ERRORS = (RuntimeError, OSError, http.client.HTTPException)


def _run_with_retry(runtime, question: str, session_id: str, tries: int = 4):
    """연결이 끊기면 몇 번 다시 시도한다. 이미 돈이 나간 호출을 살리기 위해서다."""
    for attempt in range(1, tries + 1):
        try:
            return runtime.run(question, session_id=f"{session_id}-{attempt}")
        except RETRY_ERRORS as exc:
            if attempt == tries:
                raise
            wait = 5 * attempt
            print(f"    재시도 {attempt}/{tries - 1} ({wait}초 뒤): {type(exc).__name__} {exc}", flush=True)
            time.sleep(wait)


def run(args: argparse.Namespace) -> int:
    global CONFIGS
    CONFIGS = AXES[args.axis]
    os.environ["MEMORY_ROOT"] = str(SEED_DIR)
    os.environ["LTM_CORPUS"] = str((DATA / "chunks.jsonl").resolve())
    sys.path.insert(0, str(APP))
    from org_agent_mvp.agent_runtime import AgentRuntime
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.mock_llm import MockLLMClient
    from org_agent_mvp.openrouter_client import OpenRouterClient
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    cases = _cases()
    qids = args.qids.split(",") if args.qids else _changed_qids(args.count)
    if not qids:
        raise SystemExit("중복 제거로 달라진 질문이 없다. 먼저 bench_tiers_planted를 돌릴 것")
    print(f"대상 질문 {len(qids)}건: {', '.join(qids)}  | 호출 {len(qids) * len(CONFIGS)}회"
          f"{' (모의)' if args.mock else ''}", flush=True)

    # 질문 하나가 끝날 때마다 저장한다. 중간에 죽어도 이미 지불한 호출이 남는다.
    suffix = "_mock" if args.mock else ""
    out = LOG_DIR / (f"llm_answers_{args.axis}{suffix}.json" if args.axis != "merge"
                     else f"llm_answers{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for qid in qids:
        case = cases[qid]
        entry = {"qid": qid, "question": case["question"], "gold_file": case["file_name"],
                 "gold_page": case["gold_page"], "runs": {}}
        for key, (label, env) in CONFIGS.items():
            os.environ.update(env)
            for name in list(sys.modules):
                if name.startswith("org_agent_mvp."):
                    del sys.modules[name]
            from org_agent_mvp.agent_runtime import AgentRuntime as Runtime
            from org_agent_mvp.config import AppConfig as Config
            from org_agent_mvp.memory_store import build_memory_store as build_store
            from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer as Analyzer

            cfg = Config.load()
            store = build_store(cfg)
            client = MockLLMClient() if args.mock else OpenRouterClient(cfg)
            runtime = Runtime(cfg, client, store, query_analyzer=Analyzer(vocabulary=store.filter_vocabulary()))
            # trace의 prefetch.sources는 문서 id 문자열 목록이라 계층·병합 여부가 없다.
            # 카드 상세는 prefetcher를 직접 불러 받는다(규칙 기반 분석기라 추가 비용이 없다).
            plan = runtime.query_analyzer.analyze(case["question"], [])
            cards = runtime.prefetcher.prefetch(plan).cards
            t0 = time.perf_counter()
            result = _run_with_retry(runtime, case["question"], f"llmcheck-{key}-{qid}")
            trace = result.get("trace", {})
            merged = sum(1 for c in cards if (c.get("source_ref") or {}).get("merged_evidence_ids"))
            entry["runs"][key] = {
                "label": label,
                "answer": (result.get("answer") or "")[:1500],
                "cards": [{"evidence_id": c.get("evidence_id"), "tier": c.get("tier"), "title": c.get("title"),
                           "merged": (c.get("source_ref") or {}).get("merged_evidence_ids", [])} for c in cards],
                "merged_cards": merged,
                "used_sources": (trace.get("prefetch") or {}).get("sources", []),
                "tokens": trace.get("tokens", {}),
                "expansion_count": trace.get("expansion_count", 0),
                "tool_calls": len(trace.get("tool_calls") or []),
                "seconds": round(time.perf_counter() - t0, 1),
            }
            row = entry["runs"][key]
            print(f"  {qid} [{label}] {row['seconds']}s 근거 {len(cards)}장 (접힘 {merged}) "
                  f"확장 {row['expansion_count']} 프롬프트 {(row['tokens'] or {}).get('prompt_tokens', 0):,} "
                  f"답변 {len(row['answer'])}자", flush=True)
        rows.append(entry)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    print("->", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--qids", help="쉼표로 구분한 질문 id (직접 고를 때)")
    ap.add_argument("--mock", action="store_true", help="모의 LLM으로 배선만 확인한다")
    ap.add_argument("--axis", choices=sorted(AXES), default="merge", help="무엇을 비교할지")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
