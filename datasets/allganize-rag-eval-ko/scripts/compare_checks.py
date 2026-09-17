"""check_chunks.py 결과 두 개를 비교한다 — 코드를 고친 뒤 나빠진 것이 없는지 본다.

    python scripts/compare_checks.py <이전.json> <이후.json>

  질문     판정이 바뀐 것 (이전 -> 이후), 정답 페이지 청크 포함률(cov_chunk) 변화
  문서     단어 회수율(doc / page)이 0.02 넘게 바뀐 것, 청크 수 변화
  요약     주요 수치 나란히
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

RANK = {"ok": 0, "recovered": 1, "page_shift": 2, "unreachable": 3, "no_chunk": 4, "lost": 5}


def main() -> int:
    before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    after = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    keys = ["chunks", "page_coverage_mean", "empty_pages_covered_by_ocr", "doc_recall_mean", "page_recall_mean",
            "cov_chunk_mean"]
    print("== 요약 ==")
    for k in keys:
        print(f"  {k:28s} {before['summary'].get(k)!s:>8} -> {after['summary'].get(k)!s:>8}")
    print("  verdict", before["summary"]["verdict"], "->", after["summary"]["verdict"])

    qb = {q["qid"]: q for q in before["questions"]}
    changes = collections.Counter()
    worse, better = [], []
    for q in after["questions"]:
        old = qb.get(q["qid"])
        if not old:
            continue
        if old["verdict"] != q["verdict"]:
            changes[(old["verdict"], q["verdict"])] += 1
            row = (q["qid"], q["context_type"], q["file_name"][:40], old["verdict"], q["verdict"], old["cov_chunk"], q["cov_chunk"])
            (worse if RANK[q["verdict"]] > RANK[old["verdict"]] else better).append(row)
        elif q["cov_chunk"] + 0.1 < old["cov_chunk"]:
            worse.append((q["qid"], q["context_type"], q["file_name"][:40], old["verdict"], q["verdict"], old["cov_chunk"], q["cov_chunk"]))
    print("\n== 질문 판정 변화 ==", dict(changes) or "없음")
    print("-- 좋아진 것 --")
    for r in better:
        print("  ", r)
    print("-- 나빠진 것 (판정이 내려갔거나 cov_chunk가 0.1 넘게 떨어짐) --")
    for r in worse:
        print("  ", r)
    if not worse:
        print("   없음")

    db = {d["file_name"]: d for d in before["documents"]}
    print("\n== 문서 회수율 변화 (0.02 초과) ==")
    for d in sorted(after["documents"], key=lambda d: d["file_name"]):
        old = db.get(d["file_name"])
        if not old or d["doc_recall"] is None or old["doc_recall"] is None:
            continue
        dd = d["doc_recall"] - old["doc_recall"]
        dp = d["page_recall"] - old["page_recall"]
        if abs(dd) > 0.02 or abs(dp) > 0.02:
            print(f"  doc {old['doc_recall']:.3f}->{d['doc_recall']:.3f}  page {old['page_recall']:.3f}->{d['page_recall']:.3f}  "
                  f"chunks {old['n_chunks']}->{d['n_chunks']}  {d['file_name'][:55]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
