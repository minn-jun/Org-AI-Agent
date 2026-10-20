"""시드 문서에서 평가 질문의 정답이 될 만한 대목을 캔다.

## 왜 필요한가

2026-10-19에 라벨 도구를 돌렸을 때 연쇄 30문항 중 23개, 답변 채점 20문항
중 14개가 **여러 문서에 걸렸다.** 원인은 기대값이 `0.823`, `720`처럼 짧고
흔해서다. 그 문항들은 "어느 문서가 정답인가"를 사람이 가려야 한다.

질문을 새로 만들 때는 그 문제를 **애초에 만들지 않을 수 있다.** 저장소
전체에서 **한 문서에만 나오는 값**을 골라 그것을 묻는 질문을 쓰면 라벨이
생성 시점에 이미 하나로 정해진다.

그래서 이 도구는 정답 후보를 고를 때 세 가지를 같이 본다.

1. **유일성** — 몇 개 문서에 나오는가. 1개인 것만 쓸모가 있다.
2. **위치** — 문서 앞에서 몇 %지점인가. 발췌 상한(6,000자) 밖에 있는
   대목을 일부러 골라야 잘림 손해가 측정에 드러난다.
3. **모양** — 수치·고유명사·결정 문장. 질문으로 만들기 쉬운 순서다.

## 쓰는 법

    python scripts/mine_question_anchors.py --root memory_seed_prentice_codex
    python scripts/mine_question_anchors.py --source-type codex_worklog --deep
    python scripts/mine_question_anchors.py --document 2026-08-14-codex --limit 40

`--deep`은 발췌 상한 밖(기본 6,000자 뒤)에 있는 대목만 보여 준다.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = ("stm", "mtm", "ltm")
HEADER_END = re.compile(r"^---\s*$", re.M)

#: 발췌 상한. 이 뒤에 있는 대목은 지금 구조에서 카드에 실리지 않는다.
EXCERPT_CHARS = 6000

#: 수치. 소수·분수·백분율·단위가 붙은 것만 본다. 맨 정수는 너무 흔하다.
NUMBER = re.compile(
    r"\b\d+\.\d{2,}\b"  # 0.823
    r"|\b\d+\s*/\s*\d+\b"  # 73/80
    r"|\b\d[\d,]*\s*(?:자|초|ms|개|건|원|토큰|배|점|차례|턴)\b"
    r"|\b\d+(?:\.\d+)?\s*%"
)

#: 고유명사. 라틴 문자로 된 제품·모델·기법 이름을 노린다. 흔한 낱말은 뺀다.
NAME = re.compile(r"\b[A-Za-z][A-Za-z0-9_.+-]{3,}\b")
NAME_STOP = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "your",
    "when",
    "then",
    "they",
    "will",
    "code",
    "file",
    "files",
    "true",
    "false",
    "none",
    "null",
    "self",
    "args",
    "data",
    "text",
    "name",
    "list",
    "dict",
    "test",
    "tests",
    "http",
    "https",
    "html",
    "json",
    "일단",
    "readme",
    "main",
    "user",
    "users",
    "print",
    "return",
    "import",
    "class",
    "python",
}

#: 결정·변경을 적은 문장. "무엇으로 정했나" 질문의 재료다.
DECISION = re.compile(
    r"[^\n.!?]{10,120}?(?:하기로|로 정했|로 바꿨|로 바꾸|결정했|결론은|택했|빼기로"
    r"|넣기로|쓰기로|버리기로|합의|확정)[^\n.!?]{0,80}"
)


def body_of(path: Path) -> str:
    """머리글을 떼고 본문만 돌려준다.

    머리글에는 `title`·`origin`·`summary`가 들어 있다. 거기서 값을 찾으면
    "문서 이름에 들어 있어서 맞았다"는 가짜 적중이 된다 — 2026-10-19에
    실제로 겪은 일이다.
    """
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        marks = list(HEADER_END.finditer(text))
        if len(marks) >= 2:
            return text[marks[1].end() :]
    return text


def head_of(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    header: dict[str, str] = {}
    if not text.startswith("---"):
        return header
    marks = list(HEADER_END.finditer(text))
    if len(marks) < 2:
        return header
    for line in text[marks[0].end() : marks[1].start()].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()
    return header


def fenced(body: str) -> list[tuple[int, int]]:
    """``` 코드블록의 구간.

    작업 기록에는 붙여 넣은 로그와 JSON이 많다. 그 안의 타임스탬프나
    변수명을 묻는 질문은 사람이 할 질문이 아니다. 구간을 걷어 낸다.
    """
    spans: list[tuple[int, int]] = []
    marks = [m.start() for m in re.finditer(r"```", body)]
    for index in range(0, len(marks) - 1, 2):
        spans.append((marks[index], marks[index + 1] + 3))
    return spans


def inside(spans: list[tuple[int, int]], position: int) -> bool:
    return any(left <= position < right for left, right in spans)


#: 이보다 자릿수가 많은 수는 타임스탬프·해시다. 질문 재료가 아니다.
MAX_DIGITS = 8


def sentence_at(body: str, start: int, end: int, span: int = 110) -> str:
    left = max(0, start - span)
    right = min(len(body), end + span)
    return re.sub(r"\s+", " ", body[left:right]).strip()


def anchors_of(body: str) -> list[tuple[str, str, int, int]]:
    """(종류, 값, 시작, 끝) 목록. 코드블록 안은 보지 않는다."""
    spans = fenced(body)
    found: list[tuple[str, str, int, int]] = []
    for match in NUMBER.finditer(body):
        value = match.group().strip()
        if inside(spans, match.start()):
            continue
        if len(re.sub(r"\D", "", value)) > MAX_DIGITS:
            continue
        found.append(("수치", value, match.start(), match.end()))
    for match in NAME.finditer(body):
        value = match.group()
        if value.lower() in NAME_STOP or value.isdigit():
            continue
        if inside(spans, match.start()):
            continue
        found.append(("이름", value, match.start(), match.end()))
    for match in DECISION.finditer(body):
        value = re.sub(r"\s+", " ", match.group()).strip()
        if inside(spans, match.start()):
            continue
        found.append(("결정", value, match.start(), match.end()))
    return found


def load(root: Path) -> list[dict]:
    docs: list[dict] = []
    for tier in TIERS:
        folder = root / tier
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".md", ".json", ".txt"}:
                continue
            body = body_of(path)
            docs.append(
                {
                    "tier": tier,
                    "document_id": path.name,
                    "body": body,
                    "chars": len(body),
                    "header": head_of(path) if path.suffix.lower() == ".md" else {},
                }
            )
    return docs


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="memory_seed_prentice_codex")
    parser.add_argument("--source-type", default=None, help="이 source_type 문서만 캔다")
    parser.add_argument("--document", default=None, help="이 조각이 든 문서만 캔다")
    parser.add_argument("--kind", default=None, choices=["수치", "이름", "결정"])
    parser.add_argument(
        "--deep",
        action="store_true",
        help=f"발췌 상한({EXCERPT_CHARS:,}자) 밖에 있는 대목만 본다",
    )
    parser.add_argument(
        "--per-doc",
        type=int,
        default=2,
        help="문서당 최대 몇 개까지 볼지(0이면 제한 없음)",
    )
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args(argv)

    root = ROOT / args.root
    if not root.is_dir():
        print(f"시드 루트가 없다: {root}")
        return 1
    docs = load(root)

    # 유일성은 **저장소 전체**를 기준으로 센다. 대상 문서만 보고 세면
    # 다른 계층에 같은 값이 있는 것을 놓친다.
    doc_count: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    per_doc: dict[str, list[tuple[str, str, int, int]]] = {}
    for doc in docs:
        found = anchors_of(doc["body"])
        per_doc[doc["document_id"]] = found
        for kind, value, _, _ in found:
            doc_count[(kind, value)].add(doc["document_id"])

    targets = []
    for doc in docs:
        if args.source_type and doc["header"].get("source_type") != args.source_type:
            continue
        if args.document and args.document not in doc["document_id"]:
            continue
        targets.append(doc)

    rows = []
    for doc in targets:
        seen: set[tuple[str, str]] = set()
        for kind, value, start, end in per_doc[doc["document_id"]]:
            if args.kind and kind != args.kind:
                continue
            if len(doc_count[(kind, value)]) != 1:
                continue  # 한 문서에만 나오는 값이 아니다
            if (kind, value) in seen:
                continue  # 같은 문서 안 재등장은 첫 자리만
            seen.add((kind, value))
            if args.deep and start < EXCERPT_CHARS:
                continue
            rows.append(
                {
                    "tier": doc["tier"],
                    "document_id": doc["document_id"],
                    "kind": kind,
                    "value": value,
                    "position": start,
                    "ratio": start / max(1, doc["chars"]),
                    "beyond": start >= EXCERPT_CHARS,
                    "quote": sentence_at(doc["body"], start, end),
                }
            )

    # 문서당 몇 개까지만 본다. 자리 순으로만 늘어놓으면 가장 긴 문서
    # 하나가 목록을 독점한다(78,534자 문서가 실제로 그랬다).
    if args.per_doc:
        kept: list[dict] = []
        count: dict[str, int] = {}
        for row in sorted(rows, key=lambda r: -r["ratio"]):
            seen_count = count.get(row["document_id"], 0)
            if seen_count >= args.per_doc:
                continue
            count[row["document_id"]] = seen_count + 1
            kept.append(row)
        rows = kept

    # 뒤쪽 대목부터 본다. 잘림 손해를 재려면 깊은 자리가 필요하다.
    rows.sort(key=lambda r: -r["ratio"])
    print(f"루트 {root.name}  문서 {len(docs)}개  대상 {len(targets)}개")
    print(f"유일 대목 {len(rows)}개 (한 문서에만 나오는 값만)")
    beyond = sum(1 for r in rows if r["beyond"])
    print(f"그중 발췌 {EXCERPT_CHARS:,}자 밖 {beyond}개\n")
    for row in rows[: args.limit]:
        flag = "밖" if row["beyond"] else "안"
        print(
            f"[{row['kind']}] {row['value'][:60]}\n"
            f"   {row['tier']}  {row['document_id'][:64]}\n"
            f"   {row['position']:,}자 지점 ({row['ratio']:.0%}) 발췌 {flag}\n"
            f"   …{row['quote'][:150]}…\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
