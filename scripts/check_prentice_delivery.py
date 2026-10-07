"""진행 과제 시드의 **전달**을 두 방향으로 확인한다. 유료 호출 없음.

왜 라우팅 측정과 따로 두는가
----------------------------
`bench_prentice_routing.py`의 `tier@1`은 **계층**이 맞았다는 뜻일 뿐이다.
문서를 맞게 가져와도 그 문서에서 **답이 있는 대목**이 전달되지 않으면
모델은 답할 수 없다. 2026-10-07 실측에서 그 간격이 드러났다 —
답이 있는 문서는 20/20 찾았는데, 전달된 글자에 답이 있는 것은 16/20이었다.

두 가지 모드
------------
    answers    답을 아는 20문항에서 "전달된 글자에 답이 있나"를 센다
    allganize  진행 과제 시드(STM·MTM·LTM) 위에 Allganize 청크를 LTM으로 **함께**
               붙이고, Allganize 질문이 여전히 정답 문서를 가져오는지 본다
               (진행 과제 자료가 방해자로 작동하는지)

실행
----
    python scripts/check_prentice_delivery.py answers
    SEED_EXCERPT_CHARS=6000 CONTEXT_MAX_EVIDENCE_CHARS=24000 \
        python scripts/check_prentice_delivery.py answers
    python scripts/check_prentice_delivery.py allganize --count 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = ROOT / "memory_seed_prentice"
ALLGANIZE = ROOT / "datasets" / "allganize-rag-eval-ko"

#: 답을 아는 문항과 기대 문자열. **실제 과제 값이라 저장소에 넣지 않는다.**
#: `datasets/prentice/answer_checks.jsonl`에서 읽는다(.gitignore).
#:
#:     {"qid": "r_01", "question": "...", "needles": ["기대 문자열", ...]}
ANSWER_CHECKS_PATH = ROOT / "datasets" / "prentice" / "answer_checks.jsonl"


def load_answer_checks() -> list[tuple[str, str, list[str]]]:
    if not ANSWER_CHECKS_PATH.exists():
        raise SystemExit(
            f"기대 답 목록이 없다: {ANSWER_CHECKS_PATH}\n"
            "실제 과제 값이라 저장소에 포함되지 않는다. 로컬에서 만들어야 한다."
        )
    rows = [
        json.loads(line)
        for line in ANSWER_CHECKS_PATH.open(encoding="utf-8")
        if line.strip()
    ]
    return [(r["qid"], r["question"], list(r["needles"])) for r in rows]


ANSWER_CHECKS = load_answer_checks()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text))


def loose(text: str) -> str:
    """공백 · 콤마 · 숫자 구분자를 지운 비교용 문자열.

    2026-10-07에 1차 자동 판정이 3건을 오탐했다. 원인이 전부 표기 차이였다.

      PDF 추출 공백  과제번호 원문에 숫자 사이 공백이 들어간다
      구분자         원문 `2025-12-15` vs 답변 `2025.12.15`
      천 단위 콤마    기대값 `200000` vs 답변 `200,000`

    셋 다 같은 값인데 문자열로는 다르다. 비교 전에 지운다.
    """
    text = re.sub(r"\s+", "", str(text)).replace(",", "")
    return re.sub(r"[.\-/]", "", text)


def build(seed_root: Path, corpus: Path | None):
    os.environ["MEMORY_ROOT"] = str(seed_root)
    if corpus:
        os.environ["LTM_CORPUS"] = str(corpus)
    else:
        os.environ.pop("LTM_CORPUS", None)

    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.context_builder import ContextBuilder
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.prefetch import MemoryPrefetcher
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    config = AppConfig.load()
    store = build_memory_store(config)
    return (
        config,
        store,
        RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary()),
        MemoryPrefetcher(
            store,
            total_top_k=config.prefetch_top_k,
            alpha=config.tier_prior_alpha,
            pool_per_tier=config.prefetch_pool_per_tier,
            cut_ratio=config.prefetch_cut_ratio,
            min_cards=config.prefetch_min_cards,
            tier_floor=config.prefetch_tier_floor,
        ),
        ContextBuilder(
            context_mode=config.context_mode,
            max_evidence_chars=config.max_evidence_chars,
        ),
    )


def run_answers() -> int:
    config, store, analyzer, prefetcher, builder = build(SEED, None)
    full = {
        str(doc.metadata.get("source_id", doc.path.name)): compact(doc.text)
        for doc in store.documents
    }
    print(
        f"시드 {config.memory_root.name}  문서 {len(store.documents)}건  "
        f"발췌 {store.excerpt_chars}자  근거 상한 {config.max_evidence_chars}자"
    )
    print()
    print(f"{'qid':<6}{'문서1위':>8}{'문서3내':>8}{'문서후보':>9}"
          f"{'답@1':>6}{'답@3':>6}{'답@후보':>8}  질문")
    print("-" * 94)
    stat = dict.fromkeys(("doc1", "doc3", "docN", "ans1", "ans3", "ansN"), 0)
    for qid, question, needles in ANSWER_CHECKS:
        plan = analyzer.analyze(question)
        cards = prefetcher.prefetch(plan).cards
        injected = builder.build_context(plan, cards, []).injected

        wanted = [loose(needle) for needle in needles]
        rank = None
        for index, card in enumerate(cards):
            source = str(card.get("source_ref", {}).get("document_id", ""))
            if any(needle in loose(full.get(source, "")) for needle in wanted):
                rank = index + 1
                break

        def has(cards_slice) -> bool:
            blob = loose(
                " ".join(str(c.get("content_excerpt", "")) for c in cards_slice)
            )
            return any(needle in blob for needle in wanted)

        row = {
            "doc1": rank == 1,
            "doc3": rank is not None and rank <= 3,
            "docN": rank is not None,
            "ans1": has(injected[:1]),
            "ans3": has(injected[:3]),
            "ansN": has(injected),
        }
        for key, value in row.items():
            stat[key] += bool(value)
        mark = {True: "O", False: "-"}
        print(
            f"{qid:<6}{mark[row['doc1']]:>8}{mark[row['doc3']]:>8}{mark[row['docN']]:>9}"
            f"{mark[row['ans1']]:>6}{mark[row['ans3']]:>6}{mark[row['ansN']]:>8}"
            f"  {question[:38]}"
        )
    total = len(ANSWER_CHECKS)
    print()
    print(f"답이 있는 문서  1위 {stat['doc1']}/{total}  3위내 {stat['doc3']}/{total}  "
          f"후보 {stat['docN']}/{total}")
    print(f"전달된 글자에 답  1위 {stat['ans1']}/{total}  3위내 {stat['ans3']}/{total}  "
          f"후보 {stat['ansN']}/{total}")
    return 0


def run_allganize(count: int, no_seed: bool = False) -> int:
    """진행 과제 시드 위에 Allganize 청크를 LTM으로 함께 붙인다.

    진행 과제 자료 51건이 **방해자**가 된다. 09월에 합성 방해자 698건으로
    확인한 것을 실제 자료로 다시 보는 셈이다.
    """
    sys.path.insert(0, str(ALLGANIZE / "scripts"))
    from answer_check import build_quota, load_questions, select  # type: ignore

    rows = select(load_questions(), build_quota(count), max_per_file=2)
    seed_root = SEED
    if no_seed:
        # 기준선: 진행 과제 시드 없이 Allganize 청크만 LTM에 붙인다.
        import tempfile

        empty = Path(tempfile.mkdtemp(prefix="empty_seed_"))
        for tier in ("stm", "mtm", "ltm"):
            (empty / tier).mkdir(parents=True, exist_ok=True)
        seed_root = empty
    config, store, analyzer, prefetcher, builder = build(
        seed_root, ALLGANIZE / "ltm" / "chunks.jsonl"
    )
    corpus = store.ltm_corpus
    assert corpus is not None, "LTM_CORPUS를 붙이지 못했다"

    seed_ids = {
        str(doc.metadata.get("source_id", doc.path.name)) for doc in store.documents
    }
    print(f"시드 {len(store.documents)}건 + Allganize 청크 {len(corpus.documents)}문서")
    print(f"문항 {len(rows)}건  발췌 {store.excerpt_chars}자  "
          f"근거 상한 {config.max_evidence_chars}자")
    print()
    print(f"{'qid':<8}{'유형':<11}{'후보':>4}{'정답1위':>8}{'정답후보':>9}"
          f"{'시드혼입':>9}  계층순서")
    print("-" * 88)
    gold_ok1 = gold_okN = 0
    seed_mixed = 0
    seed_cards_total = 0
    for row in rows:
        plan = analyzer.analyze(row["question"])
        cards = prefetcher.prefetch(plan).cards
        gold = Path(str(row.get("file_name") or "")).name
        names = [
            Path(str((c.get("source_ref") or {}).get("path", ""))).name for c in cards
        ]
        sources = [str((c.get("source_ref") or {}).get("document_id", "")) for c in cards]
        from_seed = [source in seed_ids for source in sources]
        rank = names.index(gold) + 1 if gold in names else None
        gold_ok1 += rank == 1
        gold_okN += rank is not None
        seed_mixed += any(from_seed)
        seed_cards_total += sum(from_seed)
        mark = {True: "O", False: "-"}
        print(
            f"{row['qid']:<8}{row['context_type']:<11}{len(cards):>4}"
            f"{mark[rank == 1]:>8}{mark[rank is not None]:>9}"
            f"{sum(from_seed):>9}  "
            + "".join(
                ("S" if seeded else str(c.get("tier", "?"))[0])
                for c, seeded in zip(cards, from_seed)
            )
        )
    total = len(rows)
    print()
    print(f"Allganize 정답 문서를 1위로   {gold_ok1}/{total}")
    print(f"Allganize 정답 문서가 후보에  {gold_okN}/{total}")
    print(f"진행 과제 시드가 섞인 문항   {seed_mixed}/{total}  "
          f"(카드 {seed_cards_total}장, 전체 후보의 "
          f"{seed_cards_total / max(sum(1 for _ in rows) * config.prefetch_top_k, 1) * 100:.0f}%)")
    print()
    print("계층순서 표기: S=진행 과제 시드 / l=Allganize LTM 청크")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("answers", "allganize"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--no-seed", action="store_true",
                        help="진행 과제 시드를 떼고 돌린다 (기준선)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.mode == "answers":
        return run_answers()
    return run_allganize(args.count, no_seed=args.no_seed)


if __name__ == "__main__":
    raise SystemExit(main())
