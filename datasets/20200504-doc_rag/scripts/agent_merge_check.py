"""Allganize에서 고른 LTM 설정을 에이전트 전체(STM/MTM/LTM 계층 병합)에 넣었을 때의 영향.

05 문서 7절 / 06 문서 5절 "다음 할 일 3". 02 문서 8절에서 BM25를 켜면 LTM과 STM/MTM의
점수 크기가 벌어져 계층 병합이 무너졌다(혼합 MRR 0.677 -> 0.429). 최종 설정에서도 그런지,
hybrid 병합(PREFETCH_NORMALIZE=hybrid)이 막아 주는지를 20200504 평가셋 60건으로 잰다.

    python agent_merge_check.py --config default      # 설정 하나씩 따로 (메모리)
    python agent_merge_check.py --config final_global_seedfreq
    ...
    python agent_merge_check.py --report              # 저장된 결과로 표 + 부트스트랩

측정은 org_agent_mvp/scripts/bench_retriever.py의 run_stage()를 그대로 쓴다(규칙 기반 analyzer, LLM 호출 없음).
결과: org_agent_mvp/logs/allganize/agent_merge_<config>.json, agent_merge_report.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

APP = Path(r"C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp")
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP))

import bench_retriever  # noqa: E402

LOG_DIR = APP / "logs" / "allganize"
KEYS = ("RETRIEVER_TOKENIZER", "RETRIEVER_SCORER", "RETRIEVER_SCORER_SEED", "RETRIEVER_BM25_K1",
        "RETRIEVER_TITLE_BONUS", "QUERY_EXPANSIONS", "RETRIEVER_DENSE", "PREFETCH_NORMALIZE", "RETRIEVER_RERANK")

#: 09-16 이전의 코드 기본값(0단계 = 공백 분리 + 빈도). 비교 기준선으로 남긴다.
#: 2026-09-16부터 코드 기본값은 아래 FINAL + 시드 BM25 + global이다.
DEFAULT = {"RETRIEVER_TOKENIZER": "whitespace", "RETRIEVER_SCORER": "freq", "RETRIEVER_SCORER_SEED": "freq",
           "RETRIEVER_BM25_K1": "2.0", "RETRIEVER_TITLE_BONUS": "add", "QUERY_EXPANSIONS": "",
           "RETRIEVER_DENSE": "0", "PREFETCH_NORMALIZE": "global", "RETRIEVER_RERANK": "none"}
#: Allganize 05에서 고른 LTM 설정 (형태소 + BM25 k1 1.2 + 제목 가산점·동의어·임베딩 없음)
FINAL = {**DEFAULT, "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_BM25_K1": "1.2",
         "RETRIEVER_TITLE_BONUS": "none", "QUERY_EXPANSIONS": "none"}

CONFIGS: dict[str, tuple[str, dict[str, str]]] = {
    "default": ("코드 기본값", DEFAULT),
    "final_global_seedfreq": ("최종 · global · 시드 빈도", {**FINAL, "RETRIEVER_SCORER_SEED": "freq", "PREFETCH_NORMALIZE": "global"}),
    "final_global_seedbm25": ("최종 · global · 시드 BM25", {**FINAL, "RETRIEVER_SCORER_SEED": "bm25", "PREFETCH_NORMALIZE": "global"}),
    "final_hybrid_seedfreq": ("최종 · hybrid · 시드 빈도", {**FINAL, "RETRIEVER_SCORER_SEED": "freq", "PREFETCH_NORMALIZE": "hybrid"}),
    "final_hybrid_seedbm25": ("최종 · hybrid · 시드 BM25", {**FINAL, "RETRIEVER_SCORER_SEED": "bm25", "PREFETCH_NORMALIZE": "hybrid"}),
    # --- 요인 분리: 02 문서 2단계(BM25 전부, k1 2.0, 제목 +2.0, 동의어 켬, global)에서 하나씩 바꾼다 ---
    "prev2_global": ("이전 2단계 · global", {**DEFAULT, "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
                                          "RETRIEVER_SCORER_SEED": "bm25"}),
    "prev2_k1": ("이전 2단계 + k1 1.2", {**DEFAULT, "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
                                     "RETRIEVER_SCORER_SEED": "bm25", "RETRIEVER_BM25_K1": "1.2"}),
    "prev2_title": ("이전 2단계 + 제목 가산점 없음", {**DEFAULT, "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
                                           "RETRIEVER_SCORER_SEED": "bm25", "RETRIEVER_TITLE_BONUS": "none"}),
    "prev2_exp": ("이전 2단계 + 동의어 없음", {**DEFAULT, "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
                                        "RETRIEVER_SCORER_SEED": "bm25", "QUERY_EXPANSIONS": "none"}),
}
#: 기본값 말고 직접 비교할 쌍 (앞 -> 뒤)
PAIRS = [("final_global_seedfreq", "final_global_seedbm25"), ("final_hybrid_seedfreq", "final_hybrid_seedbm25"),
         ("final_global_seedbm25", "final_hybrid_seedbm25"), ("prev2_global", "final_global_seedbm25")]
GROUPS = ("STM 단독", "MTM 단독", "LTM 단독", "계층 혼합")


def run_one(name: str) -> None:
    label, env = CONFIGS[name]
    assert set(env) == set(KEYS)
    bench_retriever.STAGES[name] = {"label": label, **env}
    import os
    os.environ.setdefault("MEMORY_ROOT", bench_retriever.DEFAULT_SEED)
    os.environ.setdefault("LTM_CORPUS", "auto")
    import time
    t0 = time.time()
    result = bench_retriever.run_stage(name, bench_retriever.DEFAULT_CASES)
    result["total_seconds"] = round(time.time() - t0, 1)
    result["tier_top_score_mean"] = _tier_scale()
    bench_retriever.print_report([result])
    print("계층별 최고 원점수 평균:", result["tier_top_score_mean"])
    out = LOG_DIR / f"agent_merge_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)


def _tier_scale() -> dict[str, float]:
    """02 문서 8-1절과 같은 확인: 질의마다 계층별 원점수(retrieval_score) 최고값을 구해 60건 평균.

    run_stage()가 방금 읽은 모듈(같은 환경변수)을 그대로 쓰고, 저장소만 다시 만든다.
    """
    import gc
    gc.collect()
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.prefetch import TIERS, prefetch_query
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    cfg = AppConfig.load()
    store = build_memory_store(cfg)
    analyzer = RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary())
    rows = [json.loads(l) for l in bench_retriever.DEFAULT_CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    sums = {t: 0.0 for t in TIERS}
    for case in rows:
        plan = analyzer.analyze(case["query"], case.get("recent_turns") or [])
        query = prefetch_query(plan)
        for tier in TIERS:
            res = store.retrieve(tier=tier, query=query, filters=plan.filters, top_k=cfg.prefetch_pool_per_tier)
            sums[tier] += max((float(c.get("retrieval_score", 0.0)) for c in res["results"]), default=0.0)
    out = {t: round(v / len(rows), 2) for t, v in sums.items()}
    if out.get("stm"):
        out["ltm_over_stm"] = round(out["ltm"] / out["stm"], 2)
    return out


def _case_groups() -> dict[str, str]:
    rows = [json.loads(l) for l in bench_retriever.DEFAULT_CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["id"]: bench_retriever._group_of(sorted(r["expected"]["gold_evidence_ids"]))
            for r in rows if r["expected"]["gold_evidence_ids"]}


def _rr(case: dict) -> float:
    return 1.0 / case["first_rank"] if case["first_rank"] else 0.0


def _boot(a: list[float], b: list[float], n: int = 1000, seed: int = 20260915) -> tuple[float, float, float]:
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(diffs) / k, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def report() -> None:
    groups = _case_groups()
    results = {}
    for name in CONFIGS:
        path = LOG_DIR / f"agent_merge_{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    if "default" not in results:
        raise SystemExit("default 결과가 없다")
    base = {c["id"]: c for c in results["default"]["per_case"]}
    rows = []
    print(f"{'설정':28s}{'재현율':>8s}{'MRR':>7s}{'Hit@1':>7s}{'Hit@3':>7s}" + "".join(f"{g:>10s}" for g in GROUPS)
          + f"{'LTM전멸':>8s}{'질의ms':>8s}")
    for name, r in results.items():
        o = r["overall"]
        cells = "".join(f"{r['by_group'][g]['mrr']:10.3f}" if g in r["by_group"] else f"{'-':>10s}" for g in GROUPS)
        ltm = r["by_group"].get("LTM 단독", {})
        print(f"{r['label']:28s}{o['recall']:7.1%}{o['mrr']:7.3f}{o['hit1']:7.1%}{o['hit3']:7.1%}{cells}"
              f"{ltm.get('miss', 0):8d}{r['query_ms']:8.1f}  최고점 {r.get('tier_top_score_mean', {})}")
        row = {"config": name, "label": r["label"], "settings": r["settings"], "overall": o,
               "by_group": r["by_group"], "query_ms": r["query_ms"], "vs_default": {}}
        if name != "default":
            cur = {c["id"]: c for c in r["per_case"]}
            for scope in ("전체", "LTM 단독", "계층 혼합"):
                ids = [i for i in groups if scope == "전체" or groups[i] == scope]
                d, lo, hi = _boot([_rr(base[i]) for i in ids], [_rr(cur[i]) for i in ids])
                row["vs_default"][scope] = {"n": len(ids), "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]}
            ids = list(groups)
            row["case_changes"] = {
                "전멸→회수": sum(1 for i in ids if not base[i]["first_rank"] and cur[i]["first_rank"]),
                "회수→전멸": sum(1 for i in ids if base[i]["first_rank"] and not cur[i]["first_rank"]),
            }
        rows.append(row)
    print("\n기본값 대비 MRR 차이 (paired bootstrap 1,000회, 95% CI)")
    for row in rows[1:]:
        cells = "  ".join(f"{k} {v['diff']:+.3f} ({v['ci'][0]:+.3f} ~ {v['ci'][1]:+.3f}, n={v['n']})"
                          for k, v in row["vs_default"].items())
        print(f"  {row['label']:28s} {cells}  | {row['case_changes']}")
    print("\n설정끼리 직접 비교 (앞 -> 뒤)")
    pairs_out = []
    for a, b in PAIRS:
        if a not in results or b not in results:
            continue
        ca = {c["id"]: c for c in results[a]["per_case"]}
        cb = {c["id"]: c for c in results[b]["per_case"]}
        cells, entry = [], {"from": a, "to": b}
        for scope in ("전체", "LTM 단독", "계층 혼합"):
            ids = [i for i in groups if scope == "전체" or groups[i] == scope]
            d, lo, hi = _boot([_rr(ca[i]) for i in ids], [_rr(cb[i]) for i in ids])
            entry[scope] = {"n": len(ids), "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)]}
            cells.append(f"{scope} {d:+.3f} ({lo:+.3f} ~ {hi:+.3f})")
        pairs_out.append(entry)
        print(f"  {results[a]['label']} -> {results[b]['label']}: " + "  ".join(cells))
    out = LOG_DIR / "agent_merge_report.json"
    out.write_text(json.dumps({"rows": rows, "pairs": pairs_out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)


def main() -> int:
    if "--report" in sys.argv:
        report()
        return 0
    if "--config" not in sys.argv:
        raise SystemExit(f"--config {'|'.join(CONFIGS)} 또는 --report")
    run_one(sys.argv[sys.argv.index("--config") + 1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
