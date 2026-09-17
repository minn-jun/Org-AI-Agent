"""2단계 — 정답이 LTM과 MTM 양쪽에 있을 때 무엇을 먼저 세우는가.

    python scripts/bench_tiers_planted.py --config planted
    python scripts/bench_tiers_planted.py --config planted_alpha0
    python scripts/bench_tiers_planted.py --report

`build_planted_notes.py`가 만든 폴더(방해자 400건 + 심은 노트 60건)를 시드로 쓴다.

## 채점

심은 질문(60개)에 대해 상위 8장에서 세 가지를 본다.

    ltm_rank      원본 LTM 문서(정답 파일)의 순위
    note_rank     심은 MTM 노트의 순위
    first         둘 중 먼저 온 쪽 (또는 없음)

심지 않은 질문(239개)은 1단계와 같은 방식으로 재서, 노트를 넣은 것이 나머지를 망치지 않는지 본다.

**이 수치는 동작 확인용이다.** 노트가 정답 페이지에서 파생했으므로 어휘가 겹쳐 BM25에 유리하다.
성능 수치로 인용하지 않는다(08 문서 4절).

결과: org_agent_mvp/logs/allganize/planted_<config>.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent.parent
LOG_DIR = APP / "logs" / "allganize"
DATA = ROOT / "ltm"
SEED_DIR = ROOT / "seed_tiers_planted"
TOP_K = 8

CONFIGS: dict[str, tuple[str, dict[str, str]]] = {
    # 출처 병합 끔 = 2026-09-16 이전 동작 (계층마다 같은 내용이 따로 올라온다)
    "planted_nomerge": ("출처 병합 끔 (이전 동작)", {"MEMORY_ROOT": str(SEED_DIR), "PREFETCH_MERGE_SAME_SOURCE": "0"}),
    "planted": ("출처 병합 켬", {"MEMORY_ROOT": str(SEED_DIR)}),
    "planted_alpha0": ("출처 병합 · tier prior 끔", {"MEMORY_ROOT": str(SEED_DIR), "TIER_PRIOR_ALPHA": "0"}),
    "planted_hybrid": ("출처 병합 · hybrid 병합", {"MEMORY_ROOT": str(SEED_DIR), "PREFETCH_NORMALIZE": "hybrid"}),
}


def _cases() -> list[dict]:
    rows = []
    for name in ("dev", "test"):
        with (DATA / f"cases_{name}.jsonl").open(encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return sorted(rows, key=lambda c: int(c["qid"].split("_")[1]))


def _has_original(card: dict, doc_id: str) -> bool:
    """이 카드가 정답 문서(LTM 원본)를 담고 있나 — 접혀 들어간 경우도 센다."""
    ref = card.get("source_ref") or {}
    if str(card.get("tier", "")).upper() == "LTM":
        ids = {str(ref.get("document_id"))} | {str(d) for d in ref.get("folded_document_ids", [])}
        if doc_id in ids:
            return True
    return f"ev_ltm_{doc_id}" in {str(e) for e in ref.get("merged_evidence_ids", [])}


def _has_note(card: dict, note_id: str) -> bool:
    """이 카드가 심은 노트를 담고 있나 — 접혀 들어간 경우도 센다."""
    ref = card.get("source_ref") or {}
    return card.get("evidence_id") == note_id or note_id in {str(e) for e in ref.get("merged_evidence_ids", [])}


def run_one(name: str) -> None:
    label, env = CONFIGS[name]
    os.environ.update(env)
    os.environ["LTM_CORPUS"] = str((DATA / "chunks.jsonl").resolve())
    sys.path.insert(0, str(APP))
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.prefetch import MemoryPrefetcher
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    manifest = json.loads((SEED_DIR / "planted_manifest.json").read_text(encoding="utf-8"))["notes"]
    cfg = AppConfig.load()
    store = build_memory_store(cfg)
    analyzer = RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary())
    prefetcher = MemoryPrefetcher(
        store, total_top_k=TOP_K, alpha=cfg.tier_prior_alpha, pool_per_tier=cfg.prefetch_pool_per_tier,
        cut_ratio=cfg.prefetch_cut_ratio, min_cards=cfg.prefetch_min_cards, tier_floor=cfg.prefetch_tier_floor,
    )

    planted_rows, plain_rows = [], []
    t0 = time.perf_counter()
    for case in _cases():
        plan = analyzer.analyze(case["question"], [])
        cards = prefetcher.prefetch(plan).cards
        ltm_rank = note_rank = merged_rank = None
        note_id = (manifest.get(case["qid"]) or {}).get("evidence_id")
        for i, card in enumerate(cards, 1):
            has_org = _has_original(card, case["doc_id"])
            has_note = bool(note_id) and _has_note(card, note_id)
            if has_org and ltm_rank is None:
                ltm_rank = i
            if has_note and note_rank is None:
                note_rank = i
            if has_org and has_note and merged_rank is None:
                merged_rank = i                       # 원본과 노트가 한 장으로 접힌 카드
        row = {"qid": case["qid"], "domain": case["domain"], "context_type": case["context_type"],
               "ltm_rank": ltm_rank, "note_rank": note_rank, "merged_rank": merged_rank,
               "cards": len(cards)}
        if note_id:
            row["first"] = ("한 장" if merged_rank
                            else "노트" if note_rank and (not ltm_rank or note_rank < ltm_rank)
                            else "원본" if ltm_rank else "없음")
            planted_rows.append(row)
        else:
            plain_rows.append(row)
    elapsed = time.perf_counter() - t0

    def mrr(rows: list[dict], key: str) -> float:
        return round(sum(1 / r[key] for r in rows if r[key]) / max(1, len(rows)), 4)

    def hit(rows: list[dict], key: str, k: int = TOP_K) -> float:
        return round(sum(1 for r in rows if r[key] and r[key] <= k) / max(1, len(rows)), 4)

    both = [r for r in planted_rows if r["ltm_rank"] and r["note_rank"]]
    result = {
        "config": name, "label": label, "env": env,
        "planted": {
            "n": len(planted_rows), "first": dict(Counter(r["first"] for r in planted_rows)),
            "ltm_mrr": mrr(planted_rows, "ltm_rank"), "note_mrr": mrr(planted_rows, "note_rank"),
            "ltm_hit": hit(planted_rows, "ltm_rank"), "note_hit": hit(planted_rows, "note_rank"),
            "both_in_top_k": len(both),
            "either_hit": round(sum(1 for r in planted_rows if r["ltm_rank"] or r["note_rank"]) / max(1, len(planted_rows)), 4),
        },
        "not_planted": {"n": len(plain_rows), "ltm_mrr": mrr(plain_rows, "ltm_rank"), "ltm_hit": hit(plain_rows, "ltm_rank")},
        "seed_docs": len(store.documents), "query_ms": round(elapsed / 299 * 1000, 1),
        "planted_rows": planted_rows, "plain_rows": plain_rows,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"planted_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    p = result["planted"]
    print(f"{label:26s} 시드 {result['seed_docs']:3d} | 심은 {p['n']}문항: 먼저 온 쪽 {p['first']} | "
          f"노트 MRR {p['note_mrr']:.3f} / 원본 MRR {p['ltm_mrr']:.3f} | 둘 다 상위 {p['both_in_top_k']} | "
          f"안 심은 {result['not_planted']['n']}문항 MRR {result['not_planted']['ltm_mrr']:.3f}")


def report() -> None:
    rows = []
    for name in CONFIGS:
        path = LOG_DIR / f"planted_{name}.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit("결과 파일이 없다")
    print(f"{'설정':26s}{'한장':>6s}{'노트먼저':>9s}{'원본먼저':>9s}{'둘다상위':>9s}{'노트MRR':>9s}{'원본MRR':>9s}"
          f"{'둘중하나':>9s}{'안심은MRR':>10s}")
    for r in rows:
        p = r["planted"]
        print(f"{r['label']:26s}{p['first'].get('한 장', 0):6d}{p['first'].get('노트', 0):9d}"
              f"{p['first'].get('원본', 0):9d}{p['both_in_top_k']:9d}{p['note_mrr']:9.3f}{p['ltm_mrr']:9.3f}"
              f"{p['either_hit']:9.1%}{r['not_planted']['ltm_mrr']:10.3f}")
    out = LOG_DIR / "planted_report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
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
