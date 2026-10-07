"""정답 페이지 본문이 **모델에게 실제로 전달되는지**만 잰다. LLM을 부르지 않는다.

답변 품질을 재는 answer_check.py와 달리 여기서는 전달 단계만 본다.
질의 분석은 규칙 기반으로 고정해서 실행마다 같은 결과가 나오게 한다 —
LLM 분석기를 쓰면 설정 비교에 분석기의 흔들림이 섞인다.

    python datasets/allganize-rag-eval-ko/scripts/evidence_delivery_check.py
    python datasets/allganize-rag-eval-ko/scripts/evidence_delivery_check.py \
        --set LTM_CHUNKS_PER_DOC=3 --set LTM_EXCERPT_CHARS=1200 \
        --set CONTEXT_MAX_EVIDENCE_CHARS=20000

채점
  문서 전달    정답 파일의 카드가 1차 컨텍스트에 들어갔는가
  페이지 전달  그 카드의 본문(대표 발췌 + 추가 대목)에 정답 페이지가 포함됐는가
  본문 일치    정답 페이지 청크 본문의 앞부분이 전달된 글자 안에 있는가

마지막 항목이 실제로 중요한 값이다. 문서를 찾아도 그 페이지의 글이
전달되지 않으면 모델은 답할 수 없다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets" / "allganize-rag-eval-ko" / "ltm"


def load_cases() -> list[dict]:
    cases: list[dict] = []
    for name in ("cases_dev.jsonl", "cases_test.jsonl"):
        path = DATA / name
        if path.exists():
            cases += [json.loads(line) for line in path.open(encoding="utf-8")]
    return cases


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--qids", help="쉼표로 구분. 없으면 이미 돌린 answer_check 문항을 쓴다."
    )
    parser.add_argument(
        "--count",
        type=int,
        help="answer_check와 같은 규칙으로 N문항을 새로 고른다 (유료 실행 전 무료 확인용)",
    )
    parser.add_argument(
        "--max-per-file", type=int, default=1, help="한 파일에서 뽑는 문항 수"
    )
    parser.add_argument("--md", action="store_true", help="표를 markdown으로 낸다.")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for pair in args.set:
        key, _, value = pair.partition("=")
        os.environ[key.strip()] = value.strip()
    os.environ["LTM_CORPUS"] = str(DATA / "chunks.jsonl")
    os.environ.setdefault("MEMORY_ROOT", str(ROOT / "memory_seed_empty"))

    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.context_builder import ContextBuilder
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.prefetch import MemoryPrefetcher
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    config = AppConfig.load()
    store = build_memory_store(config)
    corpus = store.ltm_corpus
    assert corpus is not None, "LTM_CORPUS를 찾지 못했다"

    analyzer = RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary())
    prefetcher = MemoryPrefetcher(
        store,
        total_top_k=config.prefetch_top_k,
        alpha=config.tier_prior_alpha,
        pool_per_tier=config.prefetch_pool_per_tier,
        cut_ratio=config.prefetch_cut_ratio,
        min_cards=config.prefetch_min_cards,
        tier_floor=config.prefetch_tier_floor,
    )
    builder = ContextBuilder(
        context_mode=config.context_mode,
        max_evidence_chars=config.max_evidence_chars,
    )

    # 정답 페이지 청크 본문을 미리 모아 둔다.
    gold_text: dict[tuple[str, int], list[str]] = {}
    for idx, doc_id in enumerate(corpus._chunk_doc):
        doc = corpus.documents[doc_id]
        for page in corpus._chunk_pages[idx]:
            gold_text.setdefault((Path(doc.rel_path).name, page), []).append(
                compact(corpus._chunk_text[idx])
            )

    cases = {case["qid"]: case for case in load_cases()}
    if args.qids:
        qids = [q.strip() for q in args.qids.split(",") if q.strip()]
    elif args.count:
        # answer_check와 **같은 선정기**를 쓴다. 두 스크립트가 다른 문항을 보면
        # 무료 측정(전달)과 유료 측정(정답)을 이어 붙일 수 없다.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from answer_check import build_quota, load_questions, select

        qids = [
            row["qid"]
            for row in select(
                load_questions(),
                build_quota(args.count),
                max_per_file=args.max_per_file,
            )
        ]
    else:
        default = ROOT / "logs" / "answer_check"
        qids = sorted(
            (p.stem for p in default.glob("q_*.json")),
            key=lambda q: int(q.split("_")[1]),
        )
    qids = [q for q in qids if q in cases]

    rows = []
    for qid in qids:
        case = cases[qid]
        plan = analyzer.analyze(case["question"])
        cards = prefetcher.prefetch(plan).cards
        built = builder.build_context(plan, cards, [])

        gold_file = Path(str(case.get("file_name") or "")).name
        gold_page = int(case.get("gold_page") or 0)

        doc_ok = False
        page_ok = False
        text_ok = False
        delivered = ""
        for card, raw in zip(built.injected, cards):
            ref = raw.get("source_ref") or {}
            if Path(str(ref.get("path", ""))).name != gold_file:
                continue
            doc_ok = True
            bodies = [str(card.get("content_excerpt") or "")]
            pages = set(ref.get("page_nos") or [])
            for extra in card.get("additional_excerpts") or []:
                bodies.append(str(extra.get("content") or ""))
                pages.update(extra.get("page_nos") or [])
            delivered = " ".join(bodies)
            page_ok = gold_page in pages
            for body in gold_text.get((gold_file, gold_page), []):
                probe = body[:60]
                if probe and probe in delivered:
                    text_ok = True
                    break
            break

        rows.append(
            {
                "qid": qid,
                "type": case.get("context_type", ""),
                "candidates": len(cards),
                "injected": len(built.injected),
                "dropped": len(built.dropped_evidence_ids),
                "chars": built.evidence_chars,
                "doc": doc_ok,
                "page": page_ok,
                "text": text_ok,
            }
        )

    mark = {True: "O", False: "-"}
    if args.md:
        print("| qid | 유형 | 후보 | 주입 | 탈락 | 근거글자 | 문서 | 페이지 | 본문 |")
        print("|---|---|---:|---:|---:|---:|---|---|---|")
        for r in rows:
            print(
                f"| {r['qid']} | {r['type']} | {r['candidates']} | {r['injected']} | "
                f"{r['dropped']} | {r['chars']:,} | {mark[r['doc']]} | "
                f"{mark[r['page']]} | {mark[r['text']]} |"
            )
    else:
        print(f"{'qid':8}{'유형':12}{'후보':>5}{'주입':>5}{'탈락':>5}{'글자':>9}  문서 페이지 본문")
        for r in rows:
            print(
                f"{r['qid']:8}{r['type']:12}{r['candidates']:5}{r['injected']:5}"
                f"{r['dropped']:5}{r['chars']:9,}   {mark[r['doc']]}    "
                f"{mark[r['page']]}     {mark[r['text']]}"
            )

    n = len(rows) or 1
    print(
        f"\nn={len(rows)}  "
        f"주입 평균 {sum(r['injected'] for r in rows)/n:.1f}/{sum(r['candidates'] for r in rows)/n:.1f}  "
        f"근거글자 평균 {sum(r['chars'] for r in rows)/n:,.0f}"
    )
    print(
        f"문서 전달 {sum(r['doc'] for r in rows)}/{len(rows)}  "
        f"페이지 전달 {sum(r['page'] for r in rows)}/{len(rows)}  "
        f"본문 일치 {sum(r['text'] for r in rows)}/{len(rows)}"
    )
    print(
        "설정: "
        f"LTM_CHUNKS_PER_DOC={config.ltm_chunks_per_doc} "
        f"LTM_EXCERPT_CHARS={config.ltm_excerpt_chars} "
        f"CONTEXT_MAX_EVIDENCE_CHARS={config.max_evidence_chars} "
        f"PREFETCH_TOP_K={config.prefetch_top_k}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
