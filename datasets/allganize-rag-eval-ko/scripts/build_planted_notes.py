"""2단계 — 정답 페이지에서 파생한 MTM 노트를 심는다 (라우팅 실험용).

    python scripts/build_planted_notes.py --count 60 --base seed_tiers_400 --out seed_tiers_planted

## 무엇을 하나

1단계(방해자)는 STM/MTM에 답이 없었다. 여기서는 **일부 질문의 정답 페이지 본문을 그대로 옮긴 MTM 노트**를 넣는다.
그러면 같은 질문에 답할 수 있는 근거가 LTM(원본 문서)과 MTM(최근 정리 노트) 양쪽에 있게 된다.
보려는 것: **에이전트가 무엇을 먼저 세우는가**, tier prior가 그 선택을 바꾸는가.

## 라벨 규칙 (기계적)

    정답 페이지 X에서 파생한 노트는, 정답 페이지가 X인 질문의 정답에 추가한다.

질문을 보지 않고 **정답 페이지 본문만** 가지고 만든다. 질문 문장에 맞춰 고쳐 쓰지 않는다.
그래도 노트 어휘가 원문에서 오므로 BM25에 유리한 편향이 남는다.
**이 수치는 동작 확인용이고 성능이라고 적지 않는다**(08 문서 4절).

## 출력

- `<out>/mtm/planted-<qid>-*.md` : 심은 노트 (기존 방해자 `--base`를 통째로 복사한 위에 얹는다)
- `<out>/planted_manifest.json`  : qid -> 노트 evidence_id·파일명·원본 문서/페이지
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ltm"
#: 노트에 옮길 본문 길이. 정답 페이지 청크를 이어 붙인다.
BODY_CHARS = 1200
#: 노트 날짜. 원본 문서(2013~2024년)보다 최신이어야 "최근 정리 노트"라는 설정이 성립한다.
NOTE_DATE = "2026-09-01"


def _cases() -> list[dict]:
    rows = []
    for name in ("dev", "test"):
        with (DATA / f"cases_{name}.jsonl").open(encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    return sorted(rows, key=lambda c: int(c["qid"].split("_")[1]))


def _gold_page_text() -> dict[tuple[str, int], list[str]]:
    """(문서 id, 페이지) -> 그 페이지에 걸친 청크 본문들."""
    out: dict[tuple[str, int], list[str]] = {}
    with (DATA / "chunks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata", {})
            pages = meta.get("page_nos") or ([meta["page_no"]] if meta.get("page_no") else [])
            for page in pages:
                out.setdefault((row["source_id"], int(page)), []).append(row["text"])
    return out


def build(count: int, base: str, out_dir: Path, seed: int) -> int:
    cases = _cases()
    pages = _gold_page_text()
    usable = [c for c in cases if (c["doc_id"], c["gold_page"]) in pages]
    picked = sorted(random.Random(seed).sample(usable, min(count, len(usable))),
                    key=lambda c: int(c["qid"].split("_")[1]))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(ROOT / base, out_dir)

    manifest = {}
    for case in picked:
        body = "\n\n".join(pages[(case["doc_id"], case["gold_page"])])[:BODY_CHARS]
        title = f"[정리] {Path(case['file_name']).stem[:50]} {case['gold_page']}쪽 발췌"
        stem = f"planted-{case['qid']}"
        path = out_dir / "mtm" / f"{stem}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        front = "\n".join([
            "---", "memory_tier: mtm", "source_type: excerpt_note", "project: 자료 정리",
            f"title: {title}", f"date: {NOTE_DATE}", "permission_scope: project_team", "status: active",
            # 출처 표기 — 계층 간 중복 제거의 판별 키다(08 문서 5-3절).
            # 이 노트는 아래 문서의 해당 쪽을 그대로 옮긴 것이라, 같은 출처의 LTM 카드와 한 장으로 접혀야 한다.
            f"source_document: {case['doc_id']}",
            f"source_page: {case['gold_page']}",
            f"summary: {re.sub(chr(10), ' ', body)[:150]}", "---",
        ])
        path.write_text(f"{front}\n# {title}\n\n{body}\n", encoding="utf-8")
        manifest[case["qid"]] = {"evidence_id": f"ev_mtm_{stem}", "file": path.name,
                                 "doc_id": case["doc_id"], "gold_page": case["gold_page"],
                                 "file_name": case["file_name"], "chars": len(body)}

    (out_dir / "planted_manifest.json").write_text(
        json.dumps({"base": base, "seed": seed, "count": len(manifest), "notes": manifest},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"심은 노트 {len(manifest)}건 (대상 후보 {len(usable)}/{len(cases)}) -> {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--base", default="seed_tiers_400", help="바탕이 될 방해자 폴더")
    ap.add_argument("--out", default="seed_tiers_planted")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()
    return build(args.count, args.base, ROOT / args.out, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
