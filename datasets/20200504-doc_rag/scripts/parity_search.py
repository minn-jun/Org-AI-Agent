"""LtmCorpus 검색 결과와 문서 속성을 덤프한다 — 과제 전용 규칙 분리 전후가 같은지 비교용.

    python parity_search.py <단계 0|1|2|3> <출력.json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP = Path(r"C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp")
DATA = Path(__file__).resolve().parents[1]
STAGES = {
    "0": {"RETRIEVER_TOKENIZER": "whitespace", "RETRIEVER_SCORER": "freq", "RETRIEVER_DENSE": "0"},
    "1": {"RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "freq", "RETRIEVER_DENSE": "0"},
    "2": {"RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_DENSE": "0"},
    "3": {"RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_DENSE": "1"},
}


def main() -> int:
    stage, out = sys.argv[1], Path(sys.argv[2])
    os.environ.update(STAGES[stage])
    os.environ["MEMORY_ROOT"] = "memory_seed_20200504"
    os.environ["LTM_CORPUS"] = "auto"
    sys.path.insert(0, str(APP))
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.ltm_corpus import LtmCorpus
    from org_agent_mvp.memory_store import _expand_query
    from org_agent_mvp.tokenizer import tokenize

    profile = next(p for p in (DATA / "corpus_profile.json", APP / "config" / "corpus_profile.json") if p.exists())
    project = json.loads(profile.read_text(encoding="utf-8"))["project"]
    corpus = LtmCorpus(AppConfig.load().ltm_corpus_path)
    cases = [json.loads(l) for l in (APP / "tests" / "fixtures" / "eval_cases_20200504.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    variants = {
        "none": None,
        "project": {"project": project},
        "other_project": {"project": "다른 과제"},
        "doctype": {"document_types": ["agreement", "plan_document"]},
    }
    rows = []
    for c in cases:
        expanded = _expand_query(c["query"])
        for name, filters in variants.items():
            res = corpus.search(tokenize(expanded), query_text=expanded, filters=filters, top_k=10)
            rows.append({"id": c["id"], "variant": name, "results": [
                [round(score, 9), card["evidence_id"], card["source_ref"]["chunk_index"],
                 card["source_ref"]["folded_document_ids"], card["source_type"], card["project"], card["title"]]
                for score, card in res]})
    docs = {d.doc_id: [d.source_type, d.family, d.final_dir, list(getattr(d, "version", None) or ())] for d in corpus.documents.values()}
    out.write_text(json.dumps({"stage": stage, "stats": corpus.stats(), "source_types": corpus.source_types(),
                               "rows": rows, "docs": docs}, ensure_ascii=False), encoding="utf-8")
    print("stage", stage, "queries", len(rows), corpus.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
