"""LTM 검색만 재서 검색기의 순수 성능을 본다.

`bench_retriever.py`는 prefetch를 거치므로 계층 병합·컷·prior가 섞인다.
검색기 자체가 좋아졌는지 보려면 그 층을 빼야 한다.

    python scripts/bench_ltm_only.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ROOT / "tests" / "fixtures" / "eval_cases_20200504.jsonl"
os.environ.setdefault("MEMORY_ROOT", "memory_seed_20200504")
os.environ.setdefault("LTM_CORPUS", "auto")

STAGES = [
    ("0 공백+빈도", {"RETRIEVER_TOKENIZER": "whitespace", "RETRIEVER_SCORER": "freq",
                  "RETRIEVER_DENSE": "0"}),
    ("1 형태소+빈도", {"RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "freq",
                   "RETRIEVER_DENSE": "0"}),
    ("2 형태소+BM25", {"RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
                    "RETRIEVER_DENSE": "0"}),
    ("3 +임베딩(RRF)", {"RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25",
                     "RETRIEVER_DENSE": "1"}),
]


def run(env: dict[str, str], top_k: int = 8) -> dict:
    for k, v in env.items():
        os.environ[k] = v
    for name in list(sys.modules):
        if name.startswith("org_agent_mvp"):
            del sys.modules[name]

    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.ltm_corpus import LtmCorpus
    from org_agent_mvp.memory_store import _expand_query
    from org_agent_mvp.tokenizer import tokenize

    corpus = LtmCorpus(AppConfig.load().ltm_corpus_path)
    rows = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [c for c in rows
             if c["expected"]["gold_evidence_ids"]
             and all(e.startswith("ev_ltm_") for e in c["expected"]["gold_evidence_ids"])]

    mrr = h1 = h3 = miss = 0
    rec = 0.0
    per_case = []
    for c in cases:
        gold = set(c["expected"]["gold_evidence_ids"])
        expanded = _expand_query(c["query"])
        res = corpus.search(tokenize(expanded), query_text=expanded, top_k=top_k)
        got = [card["evidence_id"] for _, card in res]
        hits = gold & set(got)
        rec += len(hits) / len(gold)
        first = next((i for i, e in enumerate(got, 1) if e in gold), None)
        if first:
            mrr += 1 / first
            h1 += first == 1
            h3 += first <= 3
        else:
            miss += 1
        per_case.append((c["id"], first))
    n = len(cases)
    return {"n": n, "recall": rec / n, "mrr": mrr / n, "hit1": h1 / n,
            "hit3": h3 / n, "miss": miss, "per_case": per_case}


def main() -> int:
    results = []
    for label, env in STAGES:
        print(f"[{label}] 실행 중...", flush=True)
        results.append((label, run(env)))

    print()
    print("LTM 검색만 (prefetch 계층 병합 제외)")
    print("=" * 66)
    print(f"{'단계':18s}{'재현율':>9s}{'MRR':>9s}{'Hit@1':>9s}{'Hit@3':>9s}{'전멸':>7s}")
    print("-" * 66)
    for label, r in results:
        print(f"{label:18s}{r['recall']:8.1%}{r['mrr']:9.3f}"
              f"{r['hit1']:8.1%}{r['hit3']:8.1%}{r['miss']:7d}")

    # 단계마다 어떤 케이스가 새로 풀리고 어떤 게 깨졌는지
    print()
    print("케이스별 첫 정답 순위 변화 (- 는 못 찾음)")
    print("-" * 66)
    ids = [cid for cid, _ in results[0][1]["per_case"]]
    ranks = {label: dict(r["per_case"]) for label, r in results}
    print(f"{'케이스':24s}" + "".join(f"{lab.split()[0]:>10s}" for lab, _ in results))
    for cid in ids:
        row = "".join(
            f"{(str(ranks[lab][cid]) if ranks[lab][cid] else '-'):>10s}"
            for lab, _ in results
        )
        print(f"{cid:24s}{row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
