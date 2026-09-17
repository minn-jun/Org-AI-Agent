"""doc_rag 청크를 실제 문서(PyMuPDF 페이지 텍스트)와 Allganize 정답에 대조한다.

    python scripts/check_chunks.py                      # 기본 인덱스 chunks/
    python scripts/check_chunks.py --index <폴더> --only-parsed

A. 실제 문서 대비 — 파싱에서 무엇이 빠졌나
   페이지 커버      청크의 page_nos에 한 번이라도 걸린 페이지 비율
   단어 회수율      PyMuPDF 페이지 텍스트의 단어(2글자 이상·숫자) 중 청크에 들어 있는 비율
                   문서 전체 청크 기준(doc)과 그 페이지로 표시된 청크 기준(page)을 따로 센다.
                   doc은 높은데 page가 낮으면 페이지 번호가 어긋난 것이다.
                   docling은 머리글·쪽번호를 빼므로 100%가 나오지 않는다.

B. Allganize 정답 대비 — 정답을 찾을 수 있는 청크가 있나
   gold_chunk       정답 페이지를 page_nos에 가진 청크가 있는가
   cov_pdf          정답 문구 단어가 PyMuPDF 정답 페이지에 들어 있는 비율 (기준선. check_gold_pages와 같은 계산)
   cov_chunk        같은 단어가 정답 페이지 청크들에 들어 있는 비율
   cov_doc          같은 단어가 문서 전체 청크에 들어 있는 비율
   판정
     ok             정답 페이지 청크가 있고 cov_chunk >= cov_pdf - 0.15
     page_shift     cov_chunk는 낮은데 cov_doc은 기준선 수준 (다른 페이지 청크로 들어감)
     lost           문서 전체 청크에서도 기준선보다 0.15 넘게 낮음 (파싱에서 사라짐)
     no_chunk       정답 페이지에 청크가 없음
     recovered      원문 페이지에 글자층이 없는데(cov_pdf≈0) 청크에서 정답 문구가 나옴 (OCR 효과)
     unreachable    글자층도 없고 청크에도 정답 문구가 없음
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\d[\d,.]*\d|\d|[가-힣A-Za-z]{2,}")
TOL = 0.15


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text).replace(",", "")


def tokens(text: str) -> list[str]:
    return list(dict.fromkeys(t.replace(",", "") for t in TOKEN.findall(text)))


def coverage(toks: list[str], hay: str) -> float:
    return sum(t in hay for t in toks) / len(toks) if toks and hay else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "chunks")
    ap.add_argument("--only-parsed", action="store_true", help="인덱스에 있는 문서의 질문만 본다(시험 빌드용)")
    ap.add_argument("--out", type=Path, default=ROOT / "checks" / "chunk_check.json")
    args = ap.parse_args()

    docs = {json.loads(l)["file_name"]: json.loads(l) for l in open(ROOT / "labels" / "documents.jsonl", encoding="utf-8")}
    questions = [json.loads(l) for l in open(ROOT / "labels" / "questions.jsonl", encoding="utf-8")]

    con = sqlite3.connect(args.index / "chunks.sqlite3")
    status = {r[0]: r[1:] for r in con.execute("SELECT file_name, status, n_chunks, error FROM documents")}
    chunks_by_file: dict[str, list[dict]] = collections.defaultdict(list)
    for file_name, text, page_no, page_nos in con.execute(
            "SELECT d.file_name, c.text, c.page_no, c.page_nos FROM chunks c JOIN documents d USING(doc_id) ORDER BY d.file_name, c.ord"):
        pages = json.loads(page_nos) if page_nos else ([page_no] if page_no else [])
        chunks_by_file[file_name].append({"text": text, "pages": pages, "norm": norm(text)})

    # ---------------------------------------------------------------- A. 문서 대비
    doc_rows = []
    page_text: dict[str, list[str]] = {}
    for name, meta in docs.items():
        if args.only_parsed and name not in status:
            continue
        with fitz.open(ROOT / meta["saved_as"]) as pdf:
            raw_pages = [p.get_text() for p in pdf]
        # 단어는 원문(공백 있음)에서 뽑고, 포함 여부만 공백을 지운 텍스트에서 본다.
        # 공백을 지운 뒤 단어를 뽑으면 한국어 문장이 한 덩어리가 되어 회수율이 크게 낮게 나온다
        # (2026-09-14 첫 전체 대조에서 B2B 문서 10쪽: 실제 89% -> 50%로 잘못 계산).
        page_text[name] = [norm(t) for t in raw_pages]
        chunks = chunks_by_file.get(name, [])
        doc_hay = "".join(c["norm"] for c in chunks)
        empty = set(meta["empty_text_page_list"])
        covered = {p for c in chunks for p in c["pages"]}
        doc_hits = doc_total = page_hits = 0
        low_pages = []
        for i, text in enumerate(raw_pages, start=1):
            if i in empty:
                continue
            toks = tokens(text)
            if not toks:
                continue
            page_hay = "".join(c["norm"] for c in chunks if i in c["pages"])
            d = sum(t in doc_hay for t in toks)
            p = sum(t in page_hay for t in toks)
            doc_hits += d
            page_hits += p
            doc_total += len(toks)
            if len(toks) >= 20 and d / len(toks) < 0.6:
                low_pages.append({"page": i, "doc_recall": round(d / len(toks), 2), "tokens": len(toks)})
        st = status.get(name, ("missing", 0, ""))
        doc_rows.append({
            "file_name": name, "domain": meta["domain"], "status": st[0], "n_chunks": st[1], "note": st[2],
            "pages": meta["pages"], "text_pages": meta["pages"] - len(empty), "empty_pages": len(empty),
            "page_coverage": round(len(covered) / meta["pages"], 3) if meta["pages"] else 0,
            "empty_pages_covered": len(covered & empty),
            "doc_recall": round(doc_hits / doc_total, 3) if doc_total else None,
            "page_recall": round(page_hits / doc_total, 3) if doc_total else None,
            "low_recall_pages": low_pages,
        })

    # ---------------------------------------------------------------- B. 정답 대비
    q_rows = []
    for q in questions:
        name = q["file_name"]
        if name not in page_text or q.get("exclude") or q["target_page_no"] is None:
            continue
        chunks = chunks_by_file.get(name, [])
        page = q["target_page_no"]
        toks = tokens(q["target_answer"])
        gold_chunks = [c for c in chunks if page in c["pages"]]
        cov_pdf = coverage(toks, page_text[name][page - 1])
        cov_chunk = coverage(toks, "".join(c["norm"] for c in gold_chunks))
        cov_doc = coverage(toks, "".join(c["norm"] for c in chunks))
        best_page, best = None, 0.0
        for c in chunks:
            s = coverage(toks, c["norm"])
            if s > best:
                best, best_page = s, c["pages"]
        if not q["gold_page_has_text"]:
            verdict = "recovered" if cov_chunk >= 0.3 or cov_doc >= 0.5 else "unreachable"
        elif not gold_chunks:
            verdict = "no_chunk"
        elif cov_chunk >= cov_pdf - TOL:
            verdict = "ok"
        elif cov_doc >= cov_pdf - TOL:
            verdict = "page_shift"
        else:
            verdict = "lost"
        q_rows.append({
            "qid": q["qid"], "domain": q["domain"], "context_type": q["context_type"], "file_name": name,
            "gold_page": page, "gold_chunks": len(gold_chunks), "cov_pdf": round(cov_pdf, 2),
            "cov_chunk": round(cov_chunk, 2), "cov_doc": round(cov_doc, 2),
            "best_chunk_pages": best_page, "best_chunk_cov": round(best, 2), "verdict": verdict,
        })

    # ---------------------------------------------------------------- 요약
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    lengths = [len(c["text"]) for cs in chunks_by_file.values() for c in cs]
    summary = {
        "documents": len(doc_rows),
        "status": dict(collections.Counter(r["status"] for r in doc_rows)),
        "chunks": sum(len(v) for v in chunks_by_file.values()),
        "chunk_chars": {"mean": round(sum(lengths) / len(lengths)) if lengths else 0,
                        "max": max(lengths, default=0),
                        "over_1300": sum(l > 1300 for l in lengths)},
        "chunks_without_page": sum(1 for cs in chunks_by_file.values() for c in cs if not c["pages"]),
        "page_coverage_mean": mean([r["page_coverage"] for r in doc_rows]),
        "empty_pages": sum(r["empty_pages"] for r in doc_rows),
        "empty_pages_covered_by_ocr": sum(r["empty_pages_covered"] for r in doc_rows),
        "doc_recall_mean": mean([r["doc_recall"] for r in doc_rows]),
        "page_recall_mean": mean([r["page_recall"] for r in doc_rows]),
        "questions": len(q_rows),
        "questions_excluded": sorted(q["qid"] for q in questions if q.get("exclude")),
        "verdict": dict(collections.Counter(r["verdict"] for r in q_rows)),
        "verdict_by_context_type": {t: dict(collections.Counter(r["verdict"] for r in q_rows if r["context_type"] == t))
                                    for t in sorted({r["context_type"] for r in q_rows})},
        "verdict_by_domain": {d: dict(collections.Counter(r["verdict"] for r in q_rows if r["domain"] == d))
                              for d in sorted({r["domain"] for r in q_rows})},
        "cov_pdf_mean": mean([r["cov_pdf"] for r in q_rows]),
        "cov_chunk_mean": mean([r["cov_chunk"] for r in q_rows]),
    }
    report = {"summary": summary, "documents": doc_rows, "questions": q_rows}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("\n-- 문서 (회수율 낮은 순) --")
    for r in sorted(doc_rows, key=lambda r: (r["doc_recall"] is None, r["doc_recall"] or 0))[:15]:
        print(f"{r['doc_recall']!s:>5} {r['page_recall']!s:>5} cov={r['page_coverage']:.2f} "
              f"empty={r['empty_pages']}/{r['pages']} ocr_cov={r['empty_pages_covered']} "
              f"chunks={r['n_chunks']} {r['status']} {r['note'][:40]} | {r['file_name'][:50]}")
    print("\n-- 문제 질문 --")
    for r in q_rows:
        if r["verdict"] not in ("ok", "recovered"):
            print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
