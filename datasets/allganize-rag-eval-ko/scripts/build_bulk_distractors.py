"""방해자 STM/MTM을 대량으로 만든다 — 20200504 과제 문서를 조직 중간기억 형태로 변환.

    python scripts/build_bulk_distractors.py --count 100 --out seed_tiers_100
    python scripts/build_bulk_distractors.py --count 400 --out seed_tiers_400

## 왜 필요한가

손으로 쓴 최신 자료 20건(seed_tiers)으로는 Allganize LTM 2,960청크를 흔들지 못한다(2026-09-16 측정, 동률).
"계층을 늘려도 안 무너진다"를 말하려면 **방해자가 실제로 경쟁할 만큼 많아야** 한다.
그래서 같은 실험을 방해자 20 / 100 / 400건으로 반복해 **어느 규모부터 정답이 밀리는지** 본다.

## 재료

20200504 과제 폴더(698문서, 26,031청크). Allganize 질문(금융·공공·의료·법률·커머스, 2013~2024년 공개문서)과
주제가 완전히 달라 **정답을 담고 있을 위험이 거의 없다**. 그래서 방해자로 쓰기 적합하다.

## 계층 배정

**본문은 원문 그대로 둔다** — 방해자는 회의록 같은 조직 문서 형식일 필요가 없다(2026-09-16 사용자 확인).
20200504의 `source_type`은 파일 확장자(pdf/hwp/pptx)라 유형으로 계층을 가를 수 없어서,
문서 id 해시로 **STM 3 : MTM 7** 비율로 고정 배정한다(같은 문서는 항상 같은 계층).
계층 자체가 실험 변수라 어느 문서가 어느 계층인지는 중요하지 않고, 비율과 재현성만 맞으면 된다.

## 주의

과제 문서라 **저장소 밖에만 둔다**(datasets/ 아래). 본문은 앞부분만 잘라 쓴다 — 시드는 문서 단위로
통째로 토큰화되므로 26,031청크를 다 넣으면 메모리가 감당하지 못한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT.parent / "20200504-doc_rag" / "export" / "knowledge_service_core_tech_20200504" / "chunks.jsonl"

#: STM에 넣을 비율. 나머지는 MTM.
STM_RATIO = 0.3
#: 문서 하나에서 가져올 본문 길이. 시드는 문서 단위로 통째로 색인된다.
BODY_CHARS = 1500


def load_documents() -> dict[str, dict]:
    """청크 jsonl을 문서 단위로 모은다.

    `read_text().splitlines()`를 쓰면 안 된다 — 본문에 들어 있는 제어문자(U+2028, \\x0b 등)에서
    줄이 더 쪼개져 JSON이 깨진다(2026-09-16 실측). 파일 객체는 `\\n`에서만 나눈다.
    """
    docs: dict[str, dict] = defaultdict(lambda: {"text": [], "meta": None})
    with CHUNKS.open(encoding="utf-8") as fh:
        rows = (line for line in fh if line.strip())
        for line in rows:
            row = json.loads(line)
            doc = docs[row["source_id"]]
            if doc["meta"] is None:
                doc["meta"] = row["metadata"]
            if sum(len(t) for t in doc["text"]) < BODY_CHARS:
                doc["text"].append(row["text"])
    return docs


def slug(text: str, fallback: str) -> str:
    out = re.sub(r"[^0-9A-Za-z가-힣]+", "-", str(text or "")).strip("-")[:60]
    return out or fallback


def build(count: int, out_dir: Path, seed: int) -> int:
    docs = load_documents()
    ids = sorted(docs)
    random.Random(seed).shuffle(ids)
    picked = ids[:count]
    n = {"stm": 0, "mtm": 0}
    for doc_id in picked:
        meta = docs[doc_id]["meta"] or {}
        body = re.sub(r"\n{3,}", "\n\n", "\n\n".join(docs[doc_id]["text"]))[:BODY_CHARS]
        digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=4).digest()
        tier = "stm" if digest[0] % 10 < STM_RATIO * 10 else "mtm"
        title = str(meta.get("title") or doc_id)
        path = out_dir / tier / f"{slug(title, doc_id)}-{doc_id[:8]}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        front = [
            "---",
            f"memory_tier: {tier}",
            f"source_type: {meta.get('source_type', 'document')}",
            f"project: {meta.get('project', '이전 과제')}",
            f"title: {title}",
            f"date: {str(meta.get('modified_at', ''))[:10]}",
            "permission_scope: project_team",
            "status: active",
            f"summary: {re.sub(chr(10), ' ', body)[:150]}",
            "---",
        ]
        path.write_text("\n".join(front) + f"\n# {title}\n\n{body}\n", encoding="utf-8")
        n[tier] += 1
    print(f"방해자 {sum(n.values())}건 -> {out_dir}  {n}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", default="seed_tiers_100")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()
    return build(args.count, ROOT / args.out, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
