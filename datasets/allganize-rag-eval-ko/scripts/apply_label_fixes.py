"""questions.jsonl의 정답 페이지를 pdfs/ 기준으로 맞춘다. 여러 번 돌려도 결과가 같다.

원칙 (2026-09-15 사용자 결정): **Allganize 원본 라벨을 따른다.**
원본이 가리키는 내용을 우리 PDF에서 같은 위치로 옮기는 것만 하고, 내용 판단으로 페이지를 바꾸지 않는다.
원본 라벨이 의심스러우면 label_note에만 남기고, 원본에 정답 페이지가 없으면 채점에서 뺀다.

필드
  target_page_no           pdfs/ 기준 정답 페이지 (int, 없으면 null)
  original_target_page_no  allganize 원본 값 (문자열 그대로)
  label_fix                페이지를 옮긴 이유 (안 옮겼으면 null)
  label_note               옮기지는 않았지만 알아둘 점
  exclude                  채점에서 빼는 이유 (빼지 않으면 null)
  gold_page_has_text       정답 페이지에 글자층이 있는가 (없으면 OCR이 있어야 찾을 수 있다)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 원본 목록은 48쪽인데 배포 파일(외교부 게시판 원본 = datalama 복제본, 47쪽 전부 동일)은 47쪽이다.
# 원본 라벨 13건이 모두 정답 내용이 있는 페이지보다 정확히 한 쪽 뒤를 가리킨다
# (q_68 다이아몬드 980 → 25쪽, q_116 교역 4억 7,471만 → 36쪽, q_69 무역수지 223·118 → 37쪽 등).
# Allganize 판본에 라벨 페이지(최소 21쪽)보다 앞에 한 쪽이 더 있었던 것으로 보고 −1로 옮긴다.
ANGOLA = "외교부-2024년 앙골라개황(저).pdf"
ANGOLA_REASON = ("원본 목록 48쪽 판본 기준 라벨. 배포 파일(외교부 원본=복제본)은 47쪽이고 "
                 "13건 모두 정답 내용이 한 쪽 앞 페이지에 있어 -1로 옮김(09-15 원본 PDF 대조)")
FIXES: dict[str, tuple[int, str]] = {}
NOTES = {
    "q_60": "pdfs/의 이 문서는 원본 목록과 같은 4쪽 보도자료(기획재정부 게시판 원본)로 교체함. "
            "datalama 복제본은 같은 이름의 61쪽 전체 보고서였음. 원본 라벨 4쪽에 답 수치가 모두 있음",
    "q_133": "원본 라벨 유지. 21쪽은 요점 요약이고 20쪽에 답 문구가 더 많음(글자 대조, 판단 보류)",
    "q_230": "원본 라벨 유지. 답 수치(59,113,647주, 13.21)가 2쪽에 없고 4쪽 본문에 있음(글자 대조, 판단 보류)",
    "q_75": "원본 정답 파일(..._업무계획.pdf)이 문서 목록에 없음. 복제본은 ..._보도자료.pdf로 연결함",
    "q_210": "글자 대조로는 다른 페이지가 높지만 4쪽 도면(도 2·도 3)이 정답이 맞음(렌더 확인)",
    "q_211": "글자 대조로는 다른 페이지가 높지만 4쪽 도면(도 2)이 정답이 맞음(렌더 확인)",
}
EXCLUDE = {
    "q_184": "원본 정답 페이지가 공란",
}


def main() -> None:
    docs = {}
    for line in open(ROOT / "labels" / "documents.jsonl", encoding="utf-8"):
        d = json.loads(line)
        docs[d["file_name"]] = d

    rows = [json.loads(line) for line in open(ROOT / "labels" / "questions.jsonl", encoding="utf-8")]
    changed = 0
    for q in rows:
        original = q.get("original_target_page_no", q["target_page_no"])
        q["original_target_page_no"] = original
        raw = str(original).strip()
        page = int(raw) if raw.isdigit() else None
        fix = None
        if q["file_name"] == ANGOLA and page is not None:
            page, fix = page - 1, ANGOLA_REASON
        if q["qid"] in FIXES:
            page, fix = FIXES[q["qid"]]
        q["target_page_no"] = page
        q["label_fix"] = fix
        q["label_note"] = NOTES.get(q["qid"])
        q["exclude"] = EXCLUDE.get(q["qid"]) or (None if page is not None else "정답 페이지 없음")
        doc = docs[q["file_name"]]
        if page is not None and not 1 <= page <= doc["pages"]:
            raise SystemExit(f"{q['qid']}: 정답 페이지 {page}가 문서 범위(1~{doc['pages']}) 밖")
        q["gold_page_has_text"] = None if page is None else page not in doc["empty_text_page_list"]
        changed += fix is not None

    with open(ROOT / "labels" / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    scored = [q for q in rows if not q["exclude"]]
    no_text = [q for q in scored if q["gold_page_has_text"] is False]
    print("questions", len(rows), "| scored", len(scored), "| moved", changed,
          "| notes", sum(q["label_note"] is not None for q in rows))
    print("excluded:", {q["qid"]: q["exclude"] for q in rows if q["exclude"]})
    print("gold page without text layer:", len(no_text),
          {t: sum(q["context_type"] == t for q in no_text) for t in ("paragraph", "table", "image", "text")})


if __name__ == "__main__":
    main()
