"""계층 라우팅 측정 — 질문이 **맞는 계층**에서 근거를 가져오는지 본다. 유료 호출 없음.

왜 이 측정인가
--------------
지금까지 STM · MTM은 전부 합성이었고, 그래서 "3계층이 효과가 있다"는 주장에
실측 근거가 없었다. 진행 중 과제 자료를 세 계층으로 올리면, **정답 라벨을
사람이 쓰지 않아도** 되는 질문 종류가 생긴다 — "이 질문은 어느 계층에서
답이 나와야 하는가"는 자료의 성격이 정하기 때문이다.

  LTM  확정된 계획 · 협약 · 조직 공통 매뉴얼
  MTM  주차별 진행 산출물
  STM  세미나 피드백 · 킥오프 회의

이 측정은 답변 품질이 아니라 **근거를 어디서 가져왔는지**만 본다. LLM을 부르지
않으므로 반복해도 비용이 0이고, 실행마다 같은 결과가 나온다(규칙 기반 분석기).

채점
----
  tier@1     1순위 근거의 계층이 기대 계층인가
  tier@3     상위 3개 안에 기대 계층이 있는가
  coverage   기대 계층이 후보 전체에 하나라도 들어왔는가
  doc        기대 문서(제목 부분 문자열)가 후보에 있는가 — 지정된 문항만
  mixed      기대 계층이 둘인 문항에서 **둘 다** 후보에 들어왔는가

실행
----
    MEMORY_ROOT=memory_seed_prentice python scripts/bench_prentice_routing.py
    python scripts/bench_prentice_routing.py --md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ROOT / "datasets" / "prentice" / "routing_cases.jsonl"
SEED = ROOT / "memory_seed_prentice"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in CASES.open(encoding="utf-8") if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md", action="store_true", help="표를 markdown으로 낸다")
    parser.add_argument(
        "--out",
        type=Path,
        help="문항별 결과를 json으로 저장한다 (분석기 판정·확률까지 포함).",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--analyzer",
        choices=["rule", "llm", "jev"],
        help="질의 분석기 (기본값은 QUERY_ANALYZER, 없으면 rule). "
             "jev/llm은 유료 호출이 문항마다 1회 발생한다.",
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for pair in args.set:
        key, _, value = pair.partition("=")
        os.environ[key.strip()] = value.strip()
    # 청크 코퍼스는 쓰지 않는다. 세 계층 모두 시드 문서다(길 B).
    os.environ.pop("LTM_CORPUS", None)
    os.environ["MEMORY_ROOT"] = os.environ.get("MEMORY_ROOT", str(SEED))

    from org_mem.builders import build_prefetcher, build_query_analyzer, describe
    from org_mem.config import AppConfig
    from org_mem.memory_store import build_memory_store

    config = AppConfig.load()
    store = build_memory_store(config)
    tiers = Counter(doc.tier for doc in store.documents)
    # 분석기를 바꿔 끼울 수 있어야 한다. 예전에는 규칙 기반을 직접 만들어서
    # 이 스크립트로는 **규칙 기반만** 측정할 수 있었다.
    analyzer = build_query_analyzer(config, store, kind=args.analyzer)
    prefetcher = build_prefetcher(config, store)

    cases = load_cases()
    rows = []
    # 문서 커버리지: 80문항을 다 돌려도 **한 번도 후보에 안 들어오는 문서**를 찾는다.
    # 라우팅이 맞아도 특정 문서가 영구히 안 보이면 그건 다른 문제다.
    seen_docs: set[str] = set()
    top1_docs: set[str] = set()
    for case in cases:
        expect = [t.lower() for t in case["expect_tiers"]]
        plan = analyzer.analyze(case["question"])
        decision = getattr(analyzer, "last_decision", {}) or {}
        cards = prefetcher.prefetch(plan).cards
        got = [str(card.get("tier", "")).lower() for card in cards]
        for card in cards:
            seen_docs.add(str(card.get("source_ref", {}).get("document_id", "")))
        if cards:
            top1_docs.add(str(cards[0].get("source_ref", {}).get("document_id", "")))
        # 제목은 파일명이라 내용어가 없다("2026-08-05_회의자료_<작성자>").
        # 기대 문서 판정은 제목 + 요약을 함께 본다.
        titles = [
            f"{card.get('title', '')} {card.get('summary', '')}" for card in cards
        ]

        want_doc = case.get("expect_doc")
        rows.append(
            {
                "qid": case["qid"],
                "expect": "+".join(expect),
                "weights": "".join(
                    f"{t[0]}{plan.memory_weights.get(t, 0):.1f} " for t in ("stm", "mtm", "ltm")
                ).strip(),
                "n": len(cards),
                "top1": got[0] if got else "-",
                "hit1": bool(got) and got[0] in expect,
                "hit3": any(t in expect for t in got[:3]),
                "coverage": all(t in got for t in expect),
                "doc": (
                    any(want_doc in title for title in titles) if want_doc else None
                ),
                "got": "".join(t[0] for t in got),
                # 분석기가 실제로 어느 계층을 골랐는지. 규칙 기반은 전부 검색하므로
                # 비어 있고, Jev는 고른 분기가 들어온다.
                "branch": (decision.get("jev") or {}).get("tier_branch", ""),
                "branch_conf": (decision.get("jev") or {}).get("branch_confidence"),
                "searched": "+".join((decision.get("applied") or {}).get("searched_tiers", [])),
                "overrides": (decision.get("overrides") or []),
                "analyzer_cost": decision.get("cost_usd"),
                "analyzer_latency": decision.get("latency_s"),
            }
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "config": describe(config, analyzer),
                    "seed": config.memory_root.name,
                    "documents": len(store.documents),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"저장: {args.out}")

    single = [r for r in rows if "+" not in r["expect"]]
    mixed = [r for r in rows if "+" in r["expect"]]
    with_doc = [r for r in rows if r["doc"] is not None]

    print(f"시드: {config.memory_root.name}  문서 {len(store.documents)}건 "
          f"(stm {tiers['stm']} / mtm {tiers['mtm']} / ltm {tiers['ltm']})")
    print(f"문항 {len(rows)}건  (단일 계층 {len(single)} · 계층 혼합 {len(mixed)})")
    print()

    header = f"{'qid':<6}{'기대':<9}{'가중치':<16}{'후보':>4}{'top1':>6}{'@1':>4}{'@3':>4}{'전부':>5}{'문서':>5}  계층순서"
    if args.md:
        print("| qid | 기대 | 가중치 | 후보 | top1 | @1 | @3 | 전부 | 문서 | 계층 순서 |")
        print("|---|---|---|---:|---|---|---|---|---|---|")
    else:
        print(header)
        print("-" * len(header))
    for r in rows:
        mark = lambda v: "O" if v else ("-" if v is False else " ")  # noqa: E731
        if args.md:
            print(
                f"| {r['qid']} | {r['expect']} | {r['weights']} | {r['n']} | {r['top1']} "
                f"| {mark(r['hit1'])} | {mark(r['hit3'])} | {mark(r['coverage'])} "
                f"| {mark(r['doc'])} | `{r['got']}` |"
            )
        else:
            print(
                f"{r['qid']:<6}{r['expect']:<9}{r['weights']:<16}{r['n']:>4}"
                f"{r['top1']:>6}{mark(r['hit1']):>4}{mark(r['hit3']):>4}"
                f"{mark(r['coverage']):>5}{mark(r['doc']):>5}  {r['got']}"
            )

    def rate(items, key):
        if not items:
            return "—"
        hit = sum(1 for item in items if item[key])
        return f"{hit}/{len(items)} ({hit / len(items) * 100:.0f}%)"

    print()
    print(f"{'전체':<12} tier@1 {rate(rows, 'hit1')}   tier@3 {rate(rows, 'hit3')}")
    print(f"{'단일 계층':<12} tier@1 {rate(single, 'hit1')}   tier@3 {rate(single, 'hit3')}")
    print(f"{'계층 혼합':<12} 둘 다 후보에 {rate(mixed, 'coverage')}")
    print(f"{'문서 지정':<12} 기대 문서 포함 {rate(with_doc, 'doc')}")
    print()
    for tier in ("stm", "mtm", "ltm"):
        group = [r for r in single if r["expect"] == tier]
        print(f"  {tier.upper()} 기대 {len(group):>2}문항  tier@1 {rate(group, 'hit1')}")

    # ── 문서 커버리지
    all_docs = {
        str(doc.metadata.get("source_id", doc.path.name)): doc.tier
        for doc in store.documents
    }
    never = sorted(name for name in all_docs if name not in seen_docs)
    never_top1 = sorted(
        name for name in all_docs if name in seen_docs and name not in top1_docs
    )
    print()
    print(f"문서 커버리지  후보에 한 번이라도 {len(seen_docs & set(all_docs))}/{len(all_docs)}"
          f"   1위가 된 적 있음 {len(top1_docs & set(all_docs))}/{len(all_docs)}")
    if never:
        print(f"  한 번도 후보에 안 들어온 문서 {len(never)}건")
        for name in never:
            print(f"    [{all_docs[name]}] {name[:70]}")
    if never_top1:
        print(f"  후보엔 들어오지만 1위가 된 적 없는 문서 {len(never_top1)}건")
        for name in never_top1[:12]:
            print(f"    [{all_docs[name]}] {name[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
