"""Allganize 299문항을 **에이전트 경로(계층 병합)** 로 돌린다 — STM/MTM이 붙어도 정답이 상위에 남는가.

    python scripts/bench_tiers.py --config ltm_only      # 설정 하나씩 따로 (메모리)
    python scripts/bench_tiers.py --config with_tiers
    python scripts/bench_tiers.py --report

## 기존 측정과 무엇이 다른가

`bench_allganize.py`는 `LtmCorpus`를 직접 부른다(계층 없음, 상위 10개).
여기서는 질의 분석 → 계층별 검색 → 정규화·tier prior 병합(`MemoryPrefetcher`)까지 거친 **상위 8장**을 본다.
LLM은 부르지 않는다(규칙 기반 분석기).

## 라벨

정답은 Allganize 원본 그대로다(파일 + 페이지). STM/MTM에는 정답이 없다(1단계 = 방해자).
그래서 **STM/MTM 카드가 정답보다 위에 오면 그건 전부 오염**이고, 별도로 센다.
다만 방해자가 실제로 답을 담고 있으면 라벨이 틀린 게 되므로, 정답을 제친 STM/MTM 카드는
`audit` 목록으로 뽑아 사람이 확인한다.

결과: org_agent_mvp/logs/allganize/tiers_<config>.json, tiers_report.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent.parent
LOG_DIR = APP / "logs" / "allganize"
DATA = ROOT / "ltm"
SEED = ROOT / "seed_tiers"
TOP_K = 8
BOOTSTRAP = 1000
BOOT_SEED = 20260916

#: 설정별 환경변수. 검색기 설정은 건드리지 않는다 — 2026-09-16 코드 기본값 그대로를 본다.
CONFIGS: dict[str, tuple[str, dict[str, str]]] = {
    "ltm_only": ("LTM만 (기준선)", {"MEMORY_ROOT": str(SEED / "_none")}),
    "with_tiers": ("STM/MTM 추가", {"MEMORY_ROOT": str(SEED)}),
    "with_tiers_alpha0": ("STM/MTM 추가 · tier prior 끔", {"MEMORY_ROOT": str(SEED), "TIER_PRIOR_ALPHA": "0"}),
    "with_tiers_hybrid": ("STM/MTM 추가 · hybrid 병합", {"MEMORY_ROOT": str(SEED), "PREFETCH_NORMALIZE": "hybrid"}),
    "with_tiers_nofloor": ("STM/MTM 추가 · 계층 최소배정 끔", {"MEMORY_ROOT": str(SEED), "PREFETCH_TIER_FLOOR": "0"}),
    # 방해자 규모를 키운 실행. build_bulk_distractors.py가 만든 폴더를 쓴다.
    "bulk_100": ("방해자 100건", {"MEMORY_ROOT": str(ROOT / "seed_tiers_100")}),
    "bulk_400": ("방해자 400건", {"MEMORY_ROOT": str(ROOT / "seed_tiers_400")}),
    "bulk_698": ("방해자 698건 (전체)", {"MEMORY_ROOT": str(ROOT / "seed_tiers_698")}),
    "bulk_400_hybrid": ("방해자 400건 · hybrid 병합",
                        {"MEMORY_ROOT": str(ROOT / "seed_tiers_400"), "PREFETCH_NORMALIZE": "hybrid"}),
}
#: 설정 사이 직접 비교 (앞 -> 뒤)
PAIRS = [("ltm_only", "with_tiers"), ("with_tiers", "with_tiers_alpha0"),
         ("with_tiers", "with_tiers_hybrid"), ("with_tiers", "with_tiers_nofloor"),
         ("ltm_only", "bulk_100"), ("ltm_only", "bulk_400"), ("ltm_only", "bulk_698"),
         ("bulk_400", "bulk_400_hybrid")]


def _cases() -> list[dict]:
    rows = []
    for name in ("dev", "test"):
        rows += [json.loads(l) for l in (DATA / f"cases_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(rows, key=lambda c: int(c["qid"].split("_")[1]))


def _doc_ids(card: dict) -> set[str]:
    ref = card.get("source_ref") or {}
    return {str(ref.get("document_id"))} | {str(d) for d in ref.get("folded_document_ids", [])}


def run_one(name: str) -> None:
    label, env = CONFIGS[name]
    os.environ.update(env)
    os.environ["LTM_CORPUS"] = str((DATA / "chunks.jsonl").resolve())
    sys.path.insert(0, str(APP))
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.prefetch import MemoryPrefetcher
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    cfg = AppConfig.load()
    store = build_memory_store(cfg)
    analyzer = RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary())
    prefetcher = MemoryPrefetcher(
        store, total_top_k=TOP_K, alpha=cfg.tier_prior_alpha, pool_per_tier=cfg.prefetch_pool_per_tier,
        cut_ratio=cfg.prefetch_cut_ratio, min_cards=cfg.prefetch_min_cards, tier_floor=cfg.prefetch_tier_floor,
    )

    rows, tier_counts, audit = [], Counter(), []
    t0 = time.perf_counter()
    for case in _cases():
        plan = analyzer.analyze(case["question"], [])
        cards = prefetcher.prefetch(plan).cards
        doc_rank = page_rank = None
        for i, card in enumerate(cards, 1):
            tier_counts[card["tier"]] += 1
            if card["tier"] != "LTM" or case["doc_id"] not in _doc_ids(card):
                continue
            doc_rank = doc_rank or i
            if page_rank is None and case["gold_page"] in (card["source_ref"].get("page_nos") or []):
                page_rank = i
        above = [c for c in cards[: (doc_rank - 1) if doc_rank else len(cards)] if c["tier"] != "LTM"]
        for card in above:                      # 정답을 제친 방해자 — 라벨 오류가 아닌지 확인용
            audit.append({"qid": case["qid"], "question": case["question"][:80],
                          "tier": card["tier"], "title": card["title"], "summary": card.get("summary", "")[:120]})
        rows.append({"qid": case["qid"], "domain": case["domain"], "context_type": case["context_type"],
                     "doc_rank": doc_rank, "page_rank": page_rank, "cards": len(cards),
                     "noise_above_gold": len(above),
                     "tiers": [c["tier"] for c in cards]})
    elapsed = time.perf_counter() - t0

    def metrics(key: str) -> dict:
        ranks = [r[key] for r in rows]
        n = len(ranks) or 1
        out = {f"hit@{k}": round(sum(1 for r in ranks if r and r <= k) / n, 4) for k in (1, 3, TOP_K)}
        out[f"mrr@{TOP_K}"] = round(sum(1 / r for r in ranks if r) / n, 4)
        return out

    result = {"config": name, "label": label, "env": env, "n": len(rows),
              "doc": metrics("doc_rank"), "page": metrics("page_rank"),
              "tier_cards": dict(tier_counts),
              "noise_above_gold_mean": round(sum(r["noise_above_gold"] for r in rows) / max(1, len(rows)), 3),
              "seconds": round(elapsed, 1), "query_ms": round(elapsed / max(1, len(rows)) * 1000, 1),
              "corpus": store.ltm_corpus.stats() if store.ltm_corpus else {},
              "seed_docs": len(store.documents), "per_case": rows, "audit": audit[:60], "audit_total": len(audit)}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"tiers_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{label:28s} 시드 {result['seed_docs']:3d}건 | 문서 MRR {result['doc'][f'mrr@{TOP_K}']:.3f} "
          f"Hit@1 {result['doc']['hit@1']:.1%} Hit@{TOP_K} {result['doc'][f'hit@{TOP_K}']:.1%} | "
          f"페이지 MRR {result['page'][f'mrr@{TOP_K}']:.3f} | 계층 카드 {dict(tier_counts)} | "
          f"정답 위 방해자 평균 {result['noise_above_gold_mean']:.2f} | {result['query_ms']:.0f}ms")


def _boot(a: list[float], b: list[float]) -> tuple[float, float, float]:
    rng = random.Random(BOOT_SEED)
    diffs = [y - x for x, y in zip(a, b)]
    k = len(diffs)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(BOOTSTRAP))
    return sum(diffs) / k, means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def report() -> None:
    results = {}
    for name in CONFIGS:
        path = LOG_DIR / f"tiers_{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise SystemExit("결과 파일이 없다")
    print(f"{'설정':30s}{'문서MRR':>8s}{'H@1':>7s}{'H@8':>7s}{'쪽MRR':>8s}{'LTM카드':>8s}{'STM':>6s}{'MTM':>6s}{'방해자':>7s}{'ms':>7s}")
    for r in results.values():
        t = r["tier_cards"]
        print(f"{r['label']:30s}{r['doc']['mrr@8']:8.3f}{r['doc']['hit@1']:7.1%}{r['doc']['hit@8']:7.1%}"
              f"{r['page']['mrr@8']:8.3f}{t.get('LTM', 0):8d}{t.get('STM', 0):6d}{t.get('MTM', 0):6d}"
              f"{r['noise_above_gold_mean']:7.2f}{r['query_ms']:7.0f}")
    print("\n직접 비교 (문서 RR, paired bootstrap 1,000회 95% CI)")
    pairs_out = []
    for a, b in PAIRS:
        if a not in results or b not in results:
            continue
        ra = {c["qid"]: c for c in results[a]["per_case"]}
        rb = {c["qid"]: c for c in results[b]["per_case"]}
        ids = sorted(ra, key=lambda q: int(q.split("_")[1]))
        rr = lambda row: 1.0 / row["doc_rank"] if row["doc_rank"] else 0.0  # noqa: E731
        d, lo, hi = _boot([rr(ra[i]) for i in ids], [rr(rb[i]) for i in ids])
        worse = sum(1 for i in ids if rr(rb[i]) < rr(ra[i]))
        better = sum(1 for i in ids if rr(rb[i]) > rr(ra[i]))
        pairs_out.append({"from": a, "to": b, "diff": round(d, 4), "ci": [round(lo, 4), round(hi, 4)],
                          "better": better, "worse": worse})
        verdict = "동률" if lo <= 0 <= hi else ("좋아짐" if d > 0 else "나빠짐")
        print(f"  {results[a]['label']} -> {results[b]['label']}: {d:+.4f} ({lo:+.4f} ~ {hi:+.4f}) "
              f"좋아짐 {better} / 나빠짐 {worse} => {verdict}")
    out = LOG_DIR / "tiers_report.json"
    out.write_text(json.dumps({"rows": list(results.values()), "pairs": pairs_out}, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
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
