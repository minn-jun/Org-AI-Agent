"""정답 페이지가 실제로 답을 담고 있는지 거칠게 검사한다.

답변(target_answer)에서 숫자와 2글자 이상 단어를 뽑아, 각 페이지 본문에 몇 %가 들어 있는지 센다.
  gold_score  정답 페이지의 포함률
  best_page   같은 문서에서 포함률이 가장 높은 페이지
정답 페이지에 글자층이 없으면(스캔·이미지) 검사할 수 없으므로 따로 센다.
의심 기준: best_page != 정답 이고 best_score - gold_score >= 0.25
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\d[\d,.]*\d|\d|[가-힣A-Za-z]{2,}")
STOP = {"있다", "있습니다", "합니다", "입니다", "그리고", "또한", "이는", "대한", "위한", "통해", "따라", "경우", "것으로", "하는", "되는", "있는", "있으며", "수", "등", "및"}
MISMATCH_DOCS = {"(240411보도자료) 재정동향 4월호.pdf", "PRP008.pdf", "외교부-2024년 앙골라개황(저).pdf",
                 "[3.22.목.석간]질병관리본부_-_대한안과학회,_눈건강_관리를_위한_9대_생활수칙_발표.pdf"}


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text).replace(",", "")


def main() -> None:
    docs = {json.loads(l)["file_name"]: json.loads(l) for l in open(ROOT / "labels" / "documents.jsonl", encoding="utf-8")}
    page_text: dict[str, list[str]] = {}
    for name, d in docs.items():
        with fitz.open(ROOT / d["saved_as"]) as doc:
            page_text[name] = [norm(p.get_text()) for p in doc]

    rows = []
    for line in open(ROOT / "labels" / "questions.jsonl", encoding="utf-8"):
        q = json.loads(line)
        pages = page_text[q["file_name"]]
        tokens = [t.replace(",", "") for t in TOKEN.findall(q["target_answer"]) if t not in STOP]
        tokens = list(dict.fromkeys(tokens))
        nums = [t for t in re.findall(r"\d+", str(q["target_page_no"]))]
        gold = int(nums[0]) if nums else None

        def score(text: str) -> float:
            return sum(1 for t in tokens if t in text) / len(tokens) if tokens and text else 0.0

        scores = [score(t) for t in pages]
        best = max(range(len(scores)), key=lambda i: scores[i]) + 1
        gold_text = pages[gold - 1] if gold else ""
        if gold is None:
            status = "no_gold_page"
        elif len(gold_text) < 30:
            status = "gold_page_no_text"
        elif best != gold and scores[best - 1] - scores[gold - 1] >= 0.25:
            status = "suspect"
        else:
            status = "ok"
        rows.append({"qid": q["qid"], "domain": q["domain"], "context_type": q["context_type"], "file_name": q["file_name"],
                     "gold": gold, "gold_score": round(scores[gold - 1], 2) if gold else None,
                     "best": best, "best_score": round(scores[best - 1], 2), "tokens": len(tokens), "status": status,
                     "question": q["question"][:60]})

    print("status:", dict(collections.Counter(r["status"] for r in rows)))
    ok = [r for r in rows if r["status"] == "ok"]
    print("ok gold_score 분포: 평균 %.2f, 0.5 미만 %d건" % (sum(r["gold_score"] for r in ok) / len(ok), sum(r["gold_score"] < 0.5 for r in ok)))
    print("status x domain:", {k: dict(v) for k, v in _cross(rows, "domain").items()})
    print("status x context_type:", {k: dict(v) for k, v in _cross(rows, "context_type").items()})
    print("\n-- suspect --")
    for r in rows:
        if r["status"] == "suspect":
            print(json.dumps(r, ensure_ascii=False))
    print("\n-- 페이지 수가 원본과 다른 문서의 질문 --")
    for r in rows:
        if r["file_name"] in MISMATCH_DOCS:
            print(json.dumps(r, ensure_ascii=False))
    print("\n-- 정답 페이지 없음 --")
    for r in rows:
        if r["status"] == "no_gold_page":
            print(json.dumps(r, ensure_ascii=False))
    (ROOT / "labels" / "label_check.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def _cross(rows, key):
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        out[r[key]][r["status"]] += 1
    return out


if __name__ == "__main__":
    main()
