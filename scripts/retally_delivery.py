"""저장된 턴 기록으로 "기대 근거가 전달됐나"를 다시 센다.

왜 필요한가. 2026-10-19까지 `evidence_text()`가 도구 결과 메시지를 **문자열
통째로** 근거에 넣고 있었다. 그 문자열에는 `source_ref.document_id`(파일명)가
들어 있어서, 기대값이 날짜인 문항이 **파일명 때문에 전달 O로 잡혔다.**

실제로 걸린 경우: "그 두 사람 작업을 비교하자는 이야기가 언제 나왔어?"의
기대값 `2026-08-27`이 `2026-08-27-...회의자료-....md`라는 파일명에 매칭됐다.
정답 문서는 STM의 08-27 세미나 피드백이고 그것은 도구 결과에 들어오지
않았다. 본문이 전달된 적이 없는데 전달 O였다.

유료 실행을 다시 하지 않고 고친 기준으로 다시 세려고 이 스크립트를 쓴다.
`logs/<run>/turns/*.json`의 `transcript`가 프롬프트 전문을 담고 있어서
오프라인으로 같은 집계를 할 수 있다.

    python scripts/retally_delivery.py ac30r_jev ac30s_jev ac30t_jev
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAINS = ROOT / "datasets" / "prentice" / "session_chains.jsonl"


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def loose(text: str) -> str:
    """공백·구두점을 지운 느슨한 비교용 문자열."""
    return re.sub(r"[\s\-_.,/]", "", text).lower()


def card_text(cards: list) -> list[str]:
    blobs: list[str] = []
    for card in cards or []:
        blobs.append(str(card.get("content_excerpt") or ""))
        blobs.append(str(card.get("quote") or ""))
        blobs.append(str(card.get("summary") or ""))
        for extra in card.get("additional_excerpts") or []:
            blobs.append(str(extra.get("content") or ""))
    return blobs


def evidence_of(transcript: list) -> tuple[str, str]:
    """(본문만 모은 근거, 파일명까지 포함한 예전 근거)를 함께 돌려준다."""
    body: list[str] = []
    legacy: list[str] = []
    for message in transcript or []:
        content = str(message.get("content", ""))
        if content.startswith("[RUNTIME_CONTEXT]"):
            raw = content.split("[RUNTIME_CONTEXT]\n", 1)[1]
            payload = json.loads(raw.split("\n[/RUNTIME_CONTEXT]", 1)[0])
            cards = payload.get("prefetched_evidence", [])
            body.extend(card_text(cards))
            legacy.extend(card_text(cards))
        elif message.get("role") == "tool":
            legacy.append(content)
            try:
                payload = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                continue
            body.extend(card_text(payload.get("results", [])))
    return compact(" ".join(body)), compact(" ".join(legacy))


def load_expectations() -> dict[str, list[str]]:
    want: dict[str, list[str]] = {}
    for line in CHAINS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            want[row["question"]] = list(row.get("expect") or [])
    return want


def main(runs: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    else:  # pragma: no cover - 구버전 콘솔
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    want = load_expectations()
    for run in runs:
        folder = ROOT / "logs" / run / "turns"
        if not folder.is_dir():
            print(f"{run}: turns 폴더가 없다 — 건너뛴다")
            continue
        rows = []
        for path in sorted(folder.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            question = str(record.get("user_query") or "")
            expect = want.get(question)
            if not expect:
                continue
            body, legacy = evidence_of(record.get("transcript") or [])
            rows.append(
                {
                    "question": question,
                    "body": any(loose(v) in loose(body) for v in expect),
                    "legacy": any(loose(v) in loose(legacy) for v in expect),
                    "expect": expect,
                }
            )
        total = len(rows)
        fixed = sum(r["body"] for r in rows)
        old = sum(r["legacy"] for r in rows)
        print(f"\n== {run}  턴 {total}개")
        print(f"   예전 집계 (파일명 포함)  기대 근거 전달 {old}/{total}")
        print(f"   고친 집계 (본문만)      기대 근거 전달 {fixed}/{total}")
        inflated = [r for r in rows if r["legacy"] and not r["body"]]
        if inflated:
            print(f"   파일명 때문에 O였던 턴 {len(inflated)}개")
            for row in inflated:
                print(f"     기대 {row['expect']}  |  {row['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["ac30r_jev", "ac30s_jev", "ac30t_jev"]))
