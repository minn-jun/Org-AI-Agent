"""청크 DB를 LTM 검색기 형식으로 내보내고, 평가 질문을 dev/test로 나눈다.

    python scripts/export_ltm_jsonl.py

출력 (ltm/)
  chunks.jsonl     LtmCorpus가 읽는 청크 (metadata: title, source_path, domain, page_no, page_nos, headings)
                   문서 메타데이터(doc_type, version_group ...)는 넣지 않는다 — 범용 검색기 그대로 잰다
  cases_dev.jsonl  채점 질문 299개 중 dev
  cases_test.jsonl 나머지 test
  split.json       분할 기준과 칸별 개수

분할: 도메인 × 근거 유형 칸마다 qid 순으로 정렬 → 고정 시드로 섞기 → 절반씩.
      칸 크기가 홀수면 남는 1개를 dev/test에 번갈아 준다. 제외 질문(labels exclude)은 뺀다.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ltm"
SEED = 20260915

#: 원본 라벨 페이지에 답 문구가 거의 없는 질문. 채점은 원본 라벨대로 하고 결과에만 표시한다 (README 4-3).
FLAGS = {
    "q_133": "원본 라벨 페이지(21)에 답 문구가 적음",
    "q_230": "원본 라벨 페이지(2)에 답 수치가 없음",
}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    docs_meta = {json.loads(l)["file_name"]: json.loads(l) for l in open(ROOT / "labels" / "documents.jsonl", encoding="utf-8")}
    con = sqlite3.connect(ROOT / "chunks" / "chunks.sqlite3")
    documents = {r[0]: r for r in con.execute(
        "SELECT doc_id, rel_path, file_name, n_chunks FROM documents WHERE status = 'ok'")}
    doc_by_file = {r[2]: r[0] for r in documents.values()}

    n = 0
    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as f:
        rows = con.execute(
            "SELECT c.doc_id, c.ord, c.text, c.headings, c.page_no, c.page_nos, d.rel_path "
            "FROM chunks c JOIN documents d USING(doc_id) WHERE d.status = 'ok' ORDER BY d.rel_path, c.ord")
        for doc_id, ord_, text, headings, page_no, page_nos, rel_path in rows:
            _, _, file_name, n_chunks = documents[doc_id]
            pages = json.loads(page_nos) if page_nos else ([page_no] if page_no else [])
            f.write(json.dumps({
                "chunk_id": f"{doc_id}-{ord_:04d}",
                "source_id": doc_id,
                "chunk_index": ord_,
                "chunk_count": n_chunks,
                "metadata": {
                    "title": Path(file_name).stem,
                    "source_path": rel_path.replace("\\", "/"),
                    "domain": docs_meta[file_name]["domain"],
                    "page_no": page_no,
                    "page_nos": pages,
                    "headings": json.loads(headings) if headings else [],
                },
                "text": text,
            }, ensure_ascii=False) + "\n")
            n += 1

    questions = [json.loads(l) for l in open(ROOT / "labels" / "questions.jsonl", encoding="utf-8")]
    cases = []
    for q in questions:
        if q.get("exclude"):
            continue
        cases.append({
            "qid": q["qid"], "domain": q["domain"], "context_type": q["context_type"],
            "question": q["question"], "file_name": q["file_name"], "doc_id": doc_by_file[q["file_name"]],
            "gold_page": q["target_page_no"], "gold_page_has_text": q["gold_page_has_text"],
            "flag": FLAGS.get(q["qid"]),
        })

    # 정답 페이지에 글자층이 없는 21건은 성능이 크게 다른 집단이라 칸에 넣는다
    # (넣지 않았을 때 dev 6 / test 15로 쏠렸다).
    cells: dict[tuple[str, str, bool], list[dict]] = defaultdict(list)
    for c in cases:
        cells[(c["domain"], c["context_type"], bool(c["gold_page_has_text"]))].append(c)
    rng = random.Random(SEED)
    dev, test = [], []
    odd = 0
    for key in sorted(cells):
        items = sorted(cells[key], key=lambda c: int(c["qid"].split("_")[1]))
        rng.shuffle(items)
        half = len(items) // 2
        if len(items) % 2:
            half += 1 if odd % 2 == 0 else 0
            odd += 1
        dev += items[:half]
        test += items[half:]
    order = lambda c: int(c["qid"].split("_")[1])  # noqa: E731
    for name, part in (("cases_dev.jsonl", dev), ("cases_test.jsonl", test)):
        with open(OUT / name, "w", encoding="utf-8") as f:
            for c in sorted(part, key=order):
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

    def counts(part):
        return {"n": len(part), "domain": dict(Counter(c["domain"] for c in part)),
                "context_type": dict(Counter(c["context_type"] for c in part)),
                "gold_page_has_text_false": sum(c["gold_page_has_text"] is False for c in part),
                "flagged": sorted(c["qid"] for c in part if c["flag"])}

    split = {"seed": SEED, "rule": "domain x context_type 칸별 섞어 절반, 홀수 칸은 dev/test 번갈아",
             "excluded": sorted(q["qid"] for q in questions if q.get("exclude")),
             "dev": counts(dev), "test": counts(test)}
    (OUT / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"chunks {n} documents {len(documents)} -> {OUT / 'chunks.jsonl'}")
    print(json.dumps(split, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
