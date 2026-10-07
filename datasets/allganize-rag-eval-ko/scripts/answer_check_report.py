"""answer_check.py 결과를 표로 모은다.

    python datasets/allganize-rag-eval-ko/scripts/answer_check_report.py
    python datasets/allganize-rag-eval-ko/scripts/answer_check_report.py --md   # 문서에 붙일 표

판정(`verdict`)은 사람이 결과 JSON에 채워 넣는다. 비어 있으면 미채점으로 센다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "logs" / "answer_check"


def load(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(out_dir.glob("q_*.json"),
                            key=lambda p: int(p.stem.split("_")[1]))]


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--md", action="store_true", help="마크다운 표로 출력")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = load(args.dir)
    if not rows:
        print(f"결과 없음: {args.dir}")
        return 1

    if args.md:
        print("| qid | 도메인 | 유형 | 정답 문서 | 호출 | 재검색 | 토큰 | 판정 |")
        print("|---|---|---|---:|---:|---:|---:|---|")
        for r in rows:
            gold = "사전" if r["gold_doc_in_prefetch"] else ("추가검색" if r["gold_doc_in_final"] else "없음")
            print(f"| {r['qid']} | {r['domain']} | {r['context_type']} | {gold} | "
                  f"{r['llm_calls']} | {r['retrieve_calls']} | {r['total_tokens']:,} | "
                  f"{r.get('verdict') or '—'} |")
        print()

    n = len(rows)
    tokens = [r["total_tokens"] or 0 for r in rows]
    calls = [r["llm_calls"] or 0 for r in rows]
    retrieves = [r["retrieve_calls"] or 0 for r in rows]
    print(f"문항 {n}")
    print(f"정답 문서가 사전 검색에 포함  {sum(r['gold_doc_in_prefetch'] for r in rows)} / {n}")
    print(f"정답 문서가 최종 근거에 포함  {sum(r['gold_doc_in_final'] for r in rows)} / {n}")
    print(f"모델 호출  평균 {mean(calls):.2f}  (분포 {dict(sorted(Counter(calls).items()))})")
    print(f"재검색     평균 {mean(retrieves):.2f}  (분포 {dict(sorted(Counter(retrieves).items()))})")
    print(f"재검색 1회 이상  {sum(1 for v in retrieves if v)} / {n}")
    print(f"토큰  평균 {mean(tokens):,.0f}  중앙값 {statistics.median(tokens):,.0f}  "
          f"합계 {sum(tokens):,}")

    print("\n유형별")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["context_type"]].append(r)
    for ctype, items in sorted(by_type.items()):
        print(f"  {ctype:<10} n={len(items):<3} "
              f"정답문서 {sum(r['gold_doc_in_final'] for r in items)}/{len(items)}  "
              f"토큰 평균 {mean([r['total_tokens'] or 0 for r in items]):,.0f}  "
              f"재검색 평균 {mean([r['retrieve_calls'] or 0 for r in items]):.2f}")

    verdicts = Counter(r.get("verdict") or "미채점" for r in rows)
    print(f"\n판정  {dict(verdicts)}")
    scored = [r for r in rows if r.get("verdict")]
    if scored:
        ok = [r for r in scored if r["verdict"] == "정답"]
        print(f"  정답률 {len(ok)}/{len(scored)}")
        found = [r for r in scored if r["gold_doc_in_final"]]
        if found:
            ok_found = [r for r in found if r["verdict"] == "정답"]
            print(f"  정답 문서를 찾은 질문의 정답률 {len(ok_found)}/{len(found)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
