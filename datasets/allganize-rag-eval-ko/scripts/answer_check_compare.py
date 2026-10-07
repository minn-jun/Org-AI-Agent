"""answer_check 두 실행을 같은 질문끼리 나란히 놓는다.

    python datasets/allganize-rag-eval-ko/scripts/answer_check_compare.py \
        logs/answer_check logs/answer_check_strict --labels 기본 엄격
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def load(out_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in out_dir.glob("q_*.json")}


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs=2, type=Path)
    parser.add_argument("--labels", nargs=2, default=["A", "B"])
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    a, b = load(args.dirs[0]), load(args.dirs[1])
    la, lb = args.labels
    qids = sorted(set(a) & set(b), key=lambda q: int(q.split("_")[1]))

    print(f"| qid | 유형 | {la} 판정 | {lb} 판정 | {la} 토큰 | {lb} 토큰 | "
          f"{la} 재검색 | {lb} 재검색 |")
    print("|---|---|---|---|---:|---:|---:|---:|")
    for qid in qids:
        ra, rb = a[qid], b[qid]
        print(f"| {qid} | {ra['context_type']} | {ra.get('verdict') or '—'} | "
              f"{rb.get('verdict') or '—'} | {ra['total_tokens']:,} | {rb['total_tokens']:,} | "
              f"{ra['retrieve_calls']} | {rb['retrieve_calls']} |")

    for label, data in ((la, a), (lb, b)):
        rows = [data[q] for q in qids]
        tokens = [r["total_tokens"] or 0 for r in rows]
        print(f"\n[{label}] n={len(rows)}  "
              f"토큰 평균 {mean(tokens):,.0f} 합계 {sum(tokens):,}  "
              f"재검색 평균 {mean([r['retrieve_calls'] or 0 for r in rows]):.2f}  "
              f"호출 평균 {mean([r['llm_calls'] or 0 for r in rows]):.2f}")
        print(f"        정답 문서 최종 근거 {sum(r['gold_doc_in_final'] for r in rows)}/{len(rows)}  "
              f"판정 {dict(Counter(r.get('verdict') or '미채점' for r in rows))}")

    changed = [(q, a[q].get("verdict"), b[q].get("verdict")) for q in qids
               if a[q].get("verdict") != b[q].get("verdict")]
    if changed:
        print("\n판정이 바뀐 질문")
        for qid, va, vb in changed:
            print(f"  {qid}: {va} → {vb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
