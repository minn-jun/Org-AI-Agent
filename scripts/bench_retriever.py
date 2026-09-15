"""검색기 단계별 성능 비교.

단계는 **누적**이다. 앞 단계에서 켠 것을 뒤에서 끄지 않는다.

    0  공백 분리   + 빈도 점수      기준선
    1  형태소 분석 + 빈도 점수      토큰화만 바뀐다
    2  형태소 분석 + BM25          점수 함수가 바뀐다
    3  형태소 분석 + BM25 + 임베딩  dense가 더해진다 (RRF 융합)

계층별로 쪼개서 본다. 전체 MRR은 STM/MTM이 섞여 희석되므로
**LTM 단독 MRR**이 실제로 검색기가 좋아졌는지를 가리키는 값이다.

    python scripts/bench_retriever.py --stage 0
    python scripts/bench_retriever.py --stage 0 1
    python scripts/bench_retriever.py --all
    python scripts/bench_retriever.py --all --json out.json

LTM 코퍼스가 필요하다. 환경변수는 스크립트가 알아서 세팅한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_CASES = ROOT / "tests" / "fixtures" / "eval_cases_20200504.jsonl"
DEFAULT_SEED = "memory_seed_20200504"

#: 단계 정의. (토크나이저, 점수 함수, dense 사용 여부)
STAGES: dict[str, dict[str, str]] = {
    "0": {"label": "공백 분리 + 빈도",
          "RETRIEVER_TOKENIZER": "whitespace", "RETRIEVER_SCORER": "freq", "RETRIEVER_SCORER_SEED": "freq",
          "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "global"},
    "1": {"label": "형태소 + 빈도",
          "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "freq", "RETRIEVER_SCORER_SEED": "freq",
          "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "global"},
    "2": {"label": "형태소 + BM25",
          "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_SCORER_SEED": "bm25",
          "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "global"},
    # 코퍼스 크기가 300배 차이나 같은 점수 함수가 양쪽에 다 맞지 않는다.
    # LTM만 BM25, STM/MTM은 빈도 그대로 두는 변형이다.
    "2s": {"label": "형태소 + BM25(LTM만)",
           "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
           "RETRIEVER_SCORER_SEED": "freq",
           "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "global"},
    "3": {"label": "형태소 + BM25 + 임베딩",
          "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_SCORER_SEED": "bm25",
          "RETRIEVER_DENSE": "1", "PREFETCH_NORMALIZE": "global"},
    # --- global x zscore 기하평균 ---
    # tier / rrf / zscore 병합 변형은 LTM을 무너뜨려 결론이 났으므로 단계에서 뺐다.
    # 모드 자체는 PREFETCH_NORMALIZE로 여전히 쓸 수 있다.
    "1h": {"label": "형태소 + 빈도 (h병합)",
           "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "freq",
           "RETRIEVER_SCORER_SEED": "freq",
           "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "hybrid"},
    "2h": {"label": "형태소 + BM25 (h병합)",
           "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
           "RETRIEVER_SCORER_SEED": "bm25",
           "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "hybrid"},
    "2sh": {"label": "형태소 + BM25(LTM만) (h병합)",
            "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
            "RETRIEVER_SCORER_SEED": "freq",
            "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "hybrid"},
    "3h": {"label": "형태소 + BM25 + 임베딩 (h병합)",
           "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
           "RETRIEVER_SCORER_SEED": "bm25",
           "RETRIEVER_DENSE": "1", "PREFETCH_NORMALIZE": "hybrid"},
    "3sh": {"label": "형태소 + BM25(LTM만) + 임베딩 (h병합)",
            "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
            "RETRIEVER_SCORER_SEED": "freq",
            "RETRIEVER_DENSE": "1", "PREFETCH_NORMALIZE": "hybrid"},
    "3sz": {"label": "형태소 + BM25(LTM만) + 임베딩 (z병합)",
            "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
            "RETRIEVER_SCORER_SEED": "freq",
            "RETRIEVER_DENSE": "1", "PREFETCH_NORMALIZE": "zscore"},
}


def _group_of(gold_ids: list[str]) -> str:
    tiers = {e.split("_")[1] for e in gold_ids}
    if not tiers:
        return "검색없음"
    if len(tiers) > 1:
        return "계층 혼합"
    return {"stm": "STM 단독", "mtm": "MTM 단독", "ltm": "LTM 단독"}[tiers.pop()]


def run_stage(stage: str, cases_path: Path) -> dict:
    """한 단계를 돌린다. 모듈을 매번 새로 읽어 환경변수가 반영되게 한다."""
    env = STAGES[stage]
    for key, value in env.items():
        if key != "label":
            os.environ[key] = value

    # 설정이 모듈 상단에서 굳는 곳이 있어 매번 새로 불러온다.
    for name in list(sys.modules):
        if name.startswith("org_agent_mvp"):
            del sys.modules[name]

    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.prefetch import MemoryPrefetcher
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    cfg = AppConfig.load()
    if not cfg.ltm_corpus_path:
        raise SystemExit("LTM 코퍼스를 찾지 못했다. datasets/20200504-doc_rag/export 경로를 확인할 것.")

    t0 = time.time()
    store = build_memory_store(cfg)
    load_s = time.time() - t0

    analyzer = RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary())
    prefetcher = MemoryPrefetcher(
        store, total_top_k=cfg.prefetch_top_k, alpha=cfg.tier_prior_alpha,
        pool_per_tier=cfg.prefetch_pool_per_tier, cut_ratio=cfg.prefetch_cut_ratio,
        min_cards=cfg.prefetch_min_cards, tier_floor=cfg.prefetch_tier_floor,
    )

    cases = [json.loads(l) for l in cases_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups: dict[str, dict] = defaultdict(
        lambda: dict(n=0, full=0, part=0, miss=0, mrr=0.0, h1=0, h3=0, recall=0.0)
    )
    per_case = []
    q0 = time.time()
    for case in cases:
        gold = set(case["expected"]["gold_evidence_ids"])
        plan = analyzer.analyze(case["query"], case.get("recent_turns") or [])
        cards = prefetcher.prefetch(plan).cards
        got = [str(c["evidence_id"]) for c in cards]
        hits = gold & set(got)
        first = next((i for i, e in enumerate(got, 1) if e in gold), None)
        per_case.append({"id": case["id"], "query": case["query"],
                         "gold": len(gold), "hit": len(hits), "first_rank": first})
        if not gold:
            continue
        g = groups[_group_of(sorted(gold))]
        g["n"] += 1
        g["recall"] += len(hits) / len(gold)
        g["full" if len(hits) == len(gold) else ("miss" if not hits else "part")] += 1
        if first:
            g["mrr"] += 1.0 / first
            g["h1"] += first == 1
            g["h3"] += first <= 3
    query_s = (time.time() - q0) / max(1, len(cases))

    total = dict(n=0, full=0, part=0, miss=0, mrr=0.0, h1=0, h3=0, recall=0.0)
    for g in groups.values():
        for k in total:
            total[k] += g[k]

    def pack(d: dict) -> dict:
        n = max(1, d["n"])
        return {"cases": d["n"], "full": d["full"], "part": d["part"], "miss": d["miss"],
                "recall": d["recall"] / n, "mrr": d["mrr"] / n,
                "hit1": d["h1"] / n, "hit3": d["h3"] / n}

    return {
        "stage": stage, "label": env["label"],
        "settings": {k: v for k, v in env.items() if k != "label"},
        "load_seconds": round(load_s, 1),
        "query_ms": round(query_s * 1000, 1),
        "corpus": store.ltm_corpus.stats() if store.ltm_corpus else {},
        "overall": pack(total),
        "by_group": {k: pack(v) for k, v in groups.items()},
        "per_case": per_case,
    }


def print_report(results: list[dict]) -> None:
    print()
    print("=" * 78)
    print("검색기 단계별 비교 — 실코퍼스 평가셋")
    print("=" * 78)
    print(f"{'단계':28s}{'재현율':>8s}{'MRR':>8s}{'Hit@1':>8s}{'Hit@3':>8s}{'로드':>8s}{'질의':>9s}")
    print("-" * 78)
    for r in results:
        o = r["overall"]
        print(f"{r['stage']} {r['label']:26s}{o['recall']:7.1%}{o['mrr']:8.3f}"
              f"{o['hit1']:7.1%}{o['hit3']:7.1%}{r['load_seconds']:7.1f}s{r['query_ms']:8.1f}ms")

    print()
    print("계층별 MRR — LTM이 실제 검색기 성능을 가리킨다")
    print("-" * 78)
    order = ["STM 단독", "MTM 단독", "LTM 단독", "계층 혼합"]
    print(f"{'단계':28s}" + "".join(f"{g:>12s}" for g in order))
    for r in results:
        cells = ""
        for g in order:
            v = r["by_group"].get(g)
            cells += f"{v['mrr']:12.3f}" if v else f"{'-':>12s}"
        print(f"{r['stage']} {r['label']:26s}{cells}")

    print()
    print("LTM 단독 회수 분포 (완전 / 부분 / 전멸)")
    print("-" * 78)
    for r in results:
        v = r["by_group"].get("LTM 단독")
        if v:
            print(f"  {r['stage']} {r['label']:26s} {v['full']:3d} / {v['part']:3d} / {v['miss']:3d}"
                  f"   (총 {v['cases']}건)")

    if len(results) > 1:
        base, last = results[0], results[-1]
        print()
        print(f"{base['stage']}단계 -> {last['stage']}단계 변화")
        print("-" * 78)
        for key, ko in [("recall", "재현율"), ("mrr", "MRR"), ("hit1", "Hit@1")]:
            b, l = base["overall"][key], last["overall"][key]
            print(f"  전체 {ko:8s} {b:7.3f} -> {l:7.3f}   ({l - b:+.3f})")
        bl = base["by_group"].get("LTM 단독")
        ll = last["by_group"].get("LTM 단독")
        if bl and ll:
            print(f"  LTM  {'MRR':8s} {bl['mrr']:7.3f} -> {ll['mrr']:7.3f}   ({ll['mrr'] - bl['mrr']:+.3f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", choices=sorted(STAGES), help="돌릴 단계")
    ap.add_argument("--all", action="store_true", help="정의된 단계 전부")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--seed", default=DEFAULT_SEED, help="MEMORY_ROOT로 쓸 시드 폴더")
    ap.add_argument("--json", type=Path, help="결과 저장 경로")
    args = ap.parse_args()

    stages = sorted(STAGES) if args.all else (args.stage or ["0"])
    if not args.cases.exists():
        raise SystemExit(f"평가셋이 없다: {args.cases}")

    os.environ.setdefault("MEMORY_ROOT", args.seed)
    os.environ.setdefault("LTM_CORPUS", "auto")

    results = []
    for stage in stages:
        print(f"[{stage}] {STAGES[stage]['label']} 실행 중...", flush=True)
        results.append(run_stage(stage, args.cases))

    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
