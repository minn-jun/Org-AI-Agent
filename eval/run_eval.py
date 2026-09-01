"""근거 선별 품질 평가.

gold 라벨 대비 정밀도와 재현율을 잰다.

analyzer는 두 가지로 돌릴 수 있고, 목적이 다르다.

    --analyzer rule (기본)
        규칙 기반으로 고정한다. LLM을 호출하지 않아 무료이고 결정적이다.
        prefetch 로직만 바꿔가며 비교할 때 쓴다. analyzer가 고정되어야
        quota / prior / alpha 차이가 분리된다.
        다만 실운영은 LLM analyzer를 쓰므로 이 수치는 실운영 값이 아니다.

    --analyzer llm
        실제 analyzer를 호출한다. 유료이고 실행마다 결과가 흔들린다.
        실운영에 가까운 값을 보고 싶을 때 쓴다.
        LLM이 실패해 규칙 기반으로 대체된 건수는 따로 보고한다.

    python eval/run_eval.py                      # 통제 비교 (기본)
    python eval/run_eval.py --alpha 0            # tier 무시 ablation
    python eval/run_eval.py --mode quota         # 구 쿼터 방식 baseline
    python eval/run_eval.py --analyzer llm --sleep 3   # 실운영 추정

`--mode quota`는 tier 가중치를 검색 자리 수로 바꾸던 이전 동작을 재현한다.
점수 함수와 query rewrite는 현재 코드를 그대로 쓰므로,
쿼터와 prior의 차이만 분리해서 볼 수 있다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from org_agent_mvp.config import AppConfig  # noqa: E402
from org_agent_mvp.memory_store import MemoryStore  # noqa: E402
from org_agent_mvp.prefetch import TIERS, MemoryPrefetcher  # noqa: E402
from org_agent_mvp.openrouter_client import OpenRouterClient  # noqa: E402
from org_agent_mvp.query_analyzer import (  # noqa: E402
    LLMQueryAnalyzer,
    QueryPlan,
    RuleBasedQueryAnalyzer,
)


DEFAULT_CASES = PROJECT_ROOT / "tests" / "fixtures" / "eval_cases.jsonl"


def route_of(plan: QueryPlan) -> str:
    if plan.answer_source == "session_only":
        return "session"
    if plan.answer_source == "direct_answer":
        return "direct"
    return "memory"


def quota_allocate(weights: dict[str, float], total: int) -> dict[str, int]:
    """이전 구현의 자리 배분 로직."""
    raw = {tier: max(0.0, weights.get(tier, 0.0)) * total for tier in TIERS}
    allocation = {tier: int(raw[tier]) for tier in TIERS}
    remaining = total - sum(allocation.values())
    order = sorted(TIERS, key=lambda tier: raw[tier] - allocation[tier], reverse=True)
    for tier in order[:remaining]:
        allocation[tier] += 1
    for tier in TIERS:
        if weights.get(tier, 0.0) > 0 and allocation[tier] == 0:
            donor = max(TIERS, key=lambda item: allocation[item])
            if allocation[donor] > 1:
                allocation[donor] -= 1
                allocation[tier] = 1
    return allocation


def quota_prefetch(
    store: MemoryStore, plan: QueryPlan, total_top_k: int
) -> list[dict[str, Any]]:
    """tier별로 자리를 나눠 미리 자르고, 가산 방식으로 재정렬한다."""
    if not plan.memory_needed:
        return []
    allocation = quota_allocate(plan.memory_weights, total_top_k)
    query = plan.query_rewrites[-1]
    collected: list[dict[str, Any]] = []
    for tier in TIERS:
        top_k = allocation[tier]
        if top_k == 0:
            continue
        result = store.retrieve(
            tier=tier, query=query, filters=plan.filters, top_k=top_k
        )
        for card in result["results"]:
            card = dict(card)
            card["final_score"] = round(
                float(card.get("retrieval_score", 0.0))
                + (plan.memory_weights[tier] * 5.0),
                3,
            )
            collected.append(card)
    deduped: dict[str, dict[str, Any]] = {}
    for card in collected:
        key = str(card["evidence_id"])
        if key not in deduped or card["final_score"] > deduped[key]["final_score"]:
            deduped[key] = card
    ranked = sorted(deduped.values(), key=lambda c: c["final_score"], reverse=True)
    return ranked[:total_top_k]


def evaluate(cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    config = AppConfig.load()
    store = MemoryStore(config.memory_root)
    if args.analyzer == "llm":
        if not config.api_key:
            raise SystemExit("OPENROUTER_API_KEY가 비어 있다. --analyzer rule로 실행한다.")
        analyzer: Any = LLMQueryAnalyzer(
            OpenRouterClient(config),
            model=config.query_analyzer_model,
            max_tokens=config.query_analyzer_max_tokens,
        )
    else:
        analyzer = RuleBasedQueryAnalyzer()
    top_k = args.top_k or config.prefetch_top_k
    prefetcher = MemoryPrefetcher(
        store,
        total_top_k=top_k,
        alpha=args.alpha if args.alpha is not None else config.tier_prior_alpha,
        pool_per_tier=config.prefetch_pool_per_tier,
        cut_ratio=config.prefetch_cut_ratio,
        min_cards=config.prefetch_min_cards,
        tier_floor=args.tier_floor if args.tier_floor is not None else config.prefetch_tier_floor,
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        expected = case["expected"]
        gold = set(expected["gold_evidence_ids"])
        plan = analyzer.analyze(case["query"], case.get("recent_turns") or [])
        fallback_used = bool(getattr(analyzer, "last_fallback_used", False))
        if args.sleep:
            time.sleep(args.sleep)

        if args.mode == "quota":
            cards = quota_prefetch(store, plan, top_k)
        else:
            cards = prefetcher.prefetch(plan).cards
        retrieved = [str(c["evidence_id"]) for c in cards]

        hits = gold & set(retrieved)
        precision = len(hits) / len(retrieved) if retrieved else None
        recall = len(hits) / len(gold) if gold else None
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "route_expected": expected["route"],
                "route_actual": route_of(plan),
                "route_ok": route_of(plan) == expected["route"],
                "project_expected": expected.get("project"),
                "project_actual": plan.filters.get("project"),
                "project_ok": (expected.get("project") or None)
                == (plan.filters.get("project") or None),
                "gold": sorted(gold),
                "retrieved": retrieved,
                "hits": sorted(hits),
                "retrieved_count": len(retrieved),
                "precision": precision,
                "recall": recall,
                "full_recall": bool(gold) and hits == gold,
                "skip_ok": (not gold) and not retrieved,
                "analyzer_fallback": fallback_used,
            }
        )

    search_rows = [r for r in rows if r["gold"]]
    skip_rows = [r for r in rows if not r["gold"]]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    summary = {
        "analyzer": args.analyzer,
        "analyzer_fallbacks": sum(1 for r in rows if r["analyzer_fallback"]),
        "mode": args.mode,
        "alpha": prefetcher.alpha,
        "top_k": top_k,
        "tier_floor": prefetcher.tier_floor,
        "cases": len(rows),
        "route_accuracy": mean([1.0 if r["route_ok"] else 0.0 for r in rows]),
        "project_accuracy": mean([1.0 if r["project_ok"] else 0.0 for r in rows]),
        "search_cases": len(search_rows),
        "precision": mean([r["precision"] for r in search_rows if r["precision"] is not None]),
        "recall": mean([r["recall"] for r in search_rows if r["recall"] is not None]),
        "full_recall_rate": mean([1.0 if r["full_recall"] else 0.0 for r in search_rows]),
        "avg_retrieved": mean([float(r["retrieved_count"]) for r in search_rows]),
        "skip_cases": len(skip_rows),
        "skip_accuracy": mean([1.0 if r["skip_ok"] else 0.0 for r in skip_rows]),
    }
    p, r = summary["precision"], summary["recall"]
    summary["f1"] = round(2 * p * r / (p + r), 4) if (p + r) else 0.0
    return {"summary": summary, "rows": rows}


def print_report(result: dict[str, Any], verbose: bool) -> None:
    s = result["summary"]
    label = "규칙 기반" if s["analyzer"] == "rule" else "LLM"
    print(f"analyzer={label}  모드={s['mode']}  alpha={s['alpha']}  "
          f"top_k={s['top_k']}  케이스={s['cases']}")
    if s["analyzer"] == "rule":
        print("  ※ analyzer를 고정한 통제 비교다. 실운영(LLM analyzer) 수치가 아니다.")
    elif s["analyzer_fallbacks"]:
        print(f"  ※ {s['analyzer_fallbacks']}건은 LLM 실패로 규칙 기반이 대신 쓰였다.")
    print("-" * 62)
    print(f"  라우팅 정확도      {s['route_accuracy']:.1%}   ({s['cases']}건)")
    print(f"  프로젝트 필터 정확도 {s['project_accuracy']:.1%}   ({s['cases']}건)")
    print(f"  검색 생략 정확도    {s['skip_accuracy']:.1%}   ({s['skip_cases']}건)")
    print("-" * 62)
    print(f"  정밀도 P@{s['top_k']}        {s['precision']:.1%}")
    print(f"  재현율 R           {s['recall']:.1%}")
    print(f"  F1                 {s['f1']:.1%}")
    print(f"  정답 전부 회수 비율  {s['full_recall_rate']:.1%}   ({s['search_cases']}건 중)")
    print(f"  평균 근거 수        {s['avg_retrieved']:.2f}건")

    if verbose:
        print("\n" + "=" * 62)
        for row in result["rows"]:
            mark = "OK " if (row["full_recall"] or row["skip_ok"]) else "MISS"
            print(f"\n[{mark}] {row['id']}  {row['query'][:44]}")
            print(
                f"       route {row['route_actual']}"
                f"{'' if row['route_ok'] else f' (기대 {row_route_expected(row)})'}"
                f" | 근거 {row['retrieved_count']}건"
                + (
                    f" | P={row['precision']:.2f} R={row['recall']:.2f}"
                    if row["precision"] is not None and row["recall"] is not None
                    else ""
                )
            )
            missed = [g for g in row["gold"] if g not in row["hits"]]
            if missed:
                print(f"       놓친 정답: {', '.join(missed)}")


def row_route_expected(row: dict[str, Any]) -> str:
    return str(row["route_expected"])


def main() -> int:
    parser = argparse.ArgumentParser(description="근거 선별 품질 평가")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--alpha", type=float, default=None, help="tier prior 계수")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["prior", "quota"],
        default="prior",
        help="prior=현재 A방식, quota=이전 자리배분 baseline",
    )
    parser.add_argument(
        "--analyzer",
        choices=["rule", "llm"],
        default="rule",
        help="rule=규칙 기반 고정(무료, 결정적), llm=실제 analyzer 호출(유료)",
    )
    parser.add_argument("--tier-floor", type=int, default=None, help="계층별 최소 보장 건수")
    parser.add_argument("--sleep", type=float, default=0.0, help="LLM 호출 사이 대기 초")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", type=Path, help="결과를 JSON으로 저장")
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = evaluate(cases, args)
    print_report(result, args.verbose)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
