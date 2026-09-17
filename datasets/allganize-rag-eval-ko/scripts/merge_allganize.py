"""datalama 복제본(페이지별 PDF tar 64개)을 원래 PDF로 합치고 원본 메타와 대조한다.

출력 (datasets/allganize-rag-eval-ko/)
  pdfs/<domain>/<file_name>   합친 PDF. 원본 파일명 그대로 -> 정답 target_file_name과 맞는다
  labels/documents.jsonl           문서 메타 (pid, domain, file_name, pages, 글자층 없는 페이지 수)
  labels/questions.jsonl           질문 300건 (qid, domain, question, target_answer, file_name, target_page_no, context_type)
  checks/merge_report.json    대조 결과
"""

from __future__ import annotations

import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "_raw"
OUT_PDF = ROOT / "pdfs"
OFFICIAL = ROOT / "original" / "official_pdfs"
INVALID = re.compile(r'[<>:"/\\|?*]')
EMPTY_PAGE_CHARS = 30  # 이보다 글자가 적으면 글자층이 없는 페이지(스캔·이미지)로 본다


def main() -> None:
    documents = []
    questions: dict[str, dict] = {}
    problems: list[str] = []

    for tar_path in sorted(RAW.glob("allganize-test-*.tar")):
        pages: dict[str, dict] = defaultdict(dict)
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                key, ext = member.name.rsplit(".", 1)
                data = tar.extractfile(member).read()
                pages[key][ext] = json.loads(data) if ext == "json" else data

        entries = sorted(pages.values(), key=lambda e: e["json"]["page_number"])
        metas = {(e["json"]["pid"], e["json"]["domain"], e["json"]["file_name"], e["json"]["pages"]) for e in entries}
        if len(metas) != 1:
            problems.append(f"{tar_path.name}: 한 tar에 문서가 여럿 {metas}")
        pid, domain, file_name, pages_declared = next(iter(metas))

        numbers = [e["json"]["page_number"] for e in entries]
        if numbers != list(range(1, len(numbers) + 1)):
            problems.append(f"{file_name}: 페이지 번호가 연속이 아님 {numbers[:5]}...")

        merged = fitz.open()
        for e in entries:
            with fitz.open("pdf", e["pdf"]) as src:
                merged.insert_pdf(src)
            for tc in e["json"]["test_cases"]:
                prev = questions.get(tc["qid"])
                row = {"qid": tc["qid"], "domain": domain, "question": tc["question"],
                       "target_answer": tc["target_answer"], "file_name": file_name,
                       "pid": tc["pid"], "target_page_no": tc["target_page_no"],
                       "context_type": tc["context_type"]}
                if prev and prev != row:
                    problems.append(f"{tc['qid']}: 페이지마다 질문 내용이 다름")
                questions[tc["qid"]] = row

        safe_name = INVALID.sub("_", file_name)
        # 복제본이 원본 목록과 다른 판본이면 기관 게시판에서 받은 원본을 쓴다 (README 4-1)
        official = OFFICIAL / safe_name
        if official.exists():
            merged.close()
            merged = fitz.open(official)
        empty_pages = [i + 1 for i, page in enumerate(merged) if len(page.get_text().strip()) < EMPTY_PAGE_CHARS]
        if safe_name != file_name:
            problems.append(f"{file_name}: 파일명에 쓸 수 없는 문자가 있어 {safe_name}로 저장")
        target = OUT_PDF / domain / safe_name
        target.parent.mkdir(parents=True, exist_ok=True)
        merged.save(target, garbage=3, deflate=True)
        page_count = merged.page_count
        merged.close()

        documents.append({"pid": pid, "domain": domain, "file_name": file_name, "saved_as": str(target.relative_to(ROOT)),
                          "pages_declared": pages_declared, "pages": page_count,
                          "empty_text_pages": len(empty_pages), "empty_text_page_list": empty_pages,
                          "bytes": target.stat().st_size, "source": "official" if official.exists() else "datalama"})

    documents.sort(key=lambda d: int(d["pid"].split("_")[1]))
    q_rows = sorted(questions.values(), key=lambda q: int(q["qid"].split("_")[1]))
    with open(ROOT / "labels" / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(ROOT / "labels" / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in q_rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # --- 원본과 대조 ---
    orig_docs = pd.read_csv(ROOT / "original" / "documents.csv")
    orig_q = pd.read_csv(ROOT / "original" / "rag_evaluation_result.csv")
    mirror_docs = pd.DataFrame(documents)
    mirror_q = pd.DataFrame(q_rows)

    page_join = mirror_docs.merge(orig_docs, on="file_name", how="outer", suffixes=("", "_orig"), indicator=True)
    doc_name_mismatch = page_join[page_join["_merge"] != "both"][["file_name", "_merge"]].to_dict("records")
    page_mismatch = page_join[(page_join["_merge"] == "both") & (page_join["pages"] != page_join["pages_orig"])][
        ["file_name", "pages", "pages_declared", "pages_orig"]].to_dict("records")
    declared_mismatch = mirror_docs[mirror_docs["pages"] != mirror_docs["pages_declared"]][["file_name", "pages", "pages_declared"]].to_dict("records")

    qj = orig_q.reset_index().merge(mirror_q, on="question", how="outer", suffixes=("_orig", ""), indicator=True)
    q_unmatched = qj[qj["_merge"] != "both"][["index", "qid", "question", "_merge"]].to_dict("records")
    both = qj[qj["_merge"] == "both"]
    changed = both[(both["target_file_name"] != both["file_name"]) |
                   (both["target_page_no_orig"].astype(str).str.strip() != both["target_page_no"].astype(str).str.strip()) |
                   (both["context_type_orig"] != both["context_type"])]
    changed_rows = changed[["qid", "target_file_name", "file_name", "target_page_no_orig", "target_page_no",
                            "context_type_orig", "context_type"]].to_dict("records")

    targets_missing = sorted(set(mirror_q["file_name"]) - set(mirror_docs["file_name"]))
    bad_page_no = mirror_q[~mirror_q["target_page_no"].astype(str).str.fullmatch(r"\d+")][["qid", "file_name", "target_page_no"]].to_dict("records")
    out_of_range = []
    pages_by_file = dict(zip(mirror_docs["file_name"], mirror_docs["pages"]))
    empty_by_file = dict(zip(mirror_docs["file_name"], mirror_docs["empty_text_page_list"]))
    gold_on_empty_page = []
    for q in q_rows:
        for p in re.findall(r"\d+", str(q["target_page_no"])):
            p = int(p)
            if q["file_name"] in pages_by_file and not 1 <= p <= pages_by_file[q["file_name"]]:
                out_of_range.append({"qid": q["qid"], "file_name": q["file_name"], "page": p})
            if p in empty_by_file.get(q["file_name"], []):
                gold_on_empty_page.append({"qid": q["qid"], "context_type": q["context_type"], "file_name": q["file_name"], "page": p})

    report = {
        "documents": len(documents), "pages_total": int(mirror_docs["pages"].sum()),
        "pages_by_domain": mirror_docs.groupby("domain")["pages"].sum().to_dict(),
        "questions": len(q_rows),
        "questions_by_domain": mirror_q["domain"].value_counts().to_dict(),
        "context_type": mirror_q["context_type"].value_counts().to_dict(),
        "untargeted_documents": sorted(set(mirror_docs["file_name"]) - set(mirror_q["file_name"])),
        "empty_text_pages_total": int(mirror_docs["empty_text_pages"].sum()),
        "docs_with_empty_text_pages": mirror_docs[mirror_docs["empty_text_pages"] > 0][["file_name", "pages", "empty_text_pages"]].to_dict("records"),
        "gold_on_empty_text_page": gold_on_empty_page,
        "problems": problems,
        "vs_original": {
            "doc_name_mismatch": doc_name_mismatch, "page_count_mismatch": page_mismatch,
            "declared_vs_actual_pages": declared_mismatch,
            "questions_unmatched_by_text": q_unmatched, "questions_changed": changed_rows,
        },
        "targets_missing_document": targets_missing, "bad_target_page_no": bad_page_no,
        "target_page_out_of_range": out_of_range,
        "pdf_bytes_total": int(mirror_docs["bytes"].sum()),
    }
    (ROOT / "checks" / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
