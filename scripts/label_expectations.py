"""기대값을 **문자열**에서 **문서 + 그 문서 안 위치**로 바꾼다.

## 왜

지금 채점은 기대 문자열이 근거 어딘가에 나타나면 O를 준다. 두 가지가 샌다.

1. **파일명·머리글에 걸린다.** 시드 문서 본문은 YAML 머리글로 자기 날짜와
   제목을 품고 있다. 그래서 기대값이 `2026-08-27`인 문항은 그 날짜 문서의
   카드가 아무거나 들어오면 O가 된다. 실제로 "그 두 사람 작업을 비교하자는
   이야기가 언제 나왔어?"가 그렇게 걸렸다 — 정답은 STM 세미나 피드백인데
   같은 날짜의 MTM 회의자료가 들어와서 O였다.
2. **카드 안에 답이 들어왔는지 모른다.** 문서가 들어왔는지까지만 센다.
   계획서는 73,797자인데 카드는 앞 6,000자(8%)만 싣는다. 문서는 왔고 답은
   오지 않은 경우를 구분할 수 없다.

## 왜 손으로 적지 않는가

채팅 로그를 새 자료로 넣으면 질문도 늘어난다. 그때마다 손으로 붙이면 전부
재작업이다. 그리고 사람이 적은 라벨로 채점하면 "내가 생각한 정답"을 기준으로
삼게 되어 순환이 된다.

그래서 **라벨을 저장소에서 뽑는다.** 기대 문자열을 전 계층 문서 본문에서
찾고, 찾은 자리와 주변 대목을 라벨로 만든다. 사람은 여러 건이 나왔을 때만
고른다. 새 자료가 들어오면 같은 도구를 다시 돌린다.

## 쓰는 법

    python scripts/label_expectations.py            # 보고만
    python scripts/label_expectations.py --write    # labels/*.jsonl 생성

`--write`는 원본 데이터셋을 건드리지 않고 `datasets/prentice/labels/`에
따로 쓴다. 채점기가 라벨 파일이 있으면 그것을 쓰고, 없으면 예전 문자열
방식으로 돈다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: 기본 시드 루트. Codex 작업기록을 합친 루트를 보려면 `--seed`로 넘긴다.
SEED = ROOT / "memory_seed_prentice"
DATA = ROOT / "datasets" / "prentice"
LABELS = DATA / "labels"

#: 본문에서 기대값을 찾을 때 머리글은 빼고 본다. 머리글에는 제목·날짜가
#: 들어 있어서 날짜 기대값이 거기에 걸린다 — 그것이 이 작업의 이유다.
HEADER_END = re.compile(r"^---\s*$.*?^---\s*$", re.MULTILINE | re.DOTALL)
#: json 시드(세미나 피드백)는 메타 키가 본문과 섞여 있다. 본문 필드만 본다.
JSON_BODY_KEYS = ("body", "summary", "text", "content")
#: 기대 대목을 얼마나 떼어 라벨에 적을지.
QUOTE_CHARS = 120


def body_of(path: Path) -> str:
    """채점에 쓸 **본문**. 머리글·메타는 뺀다."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        parts = [
            str(payload.get(key) or "")
            for key in JSON_BODY_KEYS
            if payload.get(key)
        ]
        return "\n".join(parts)
    return HEADER_END.sub("", raw, count=1)


def loose(text: str) -> str:
    return re.sub(r"[\s\-_.,/]", "", text).lower()


def load_documents() -> list[dict]:
    docs = []
    for tier in ("stm", "mtm", "ltm"):
        folder = SEED / tier
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*")):
            if path.is_file():
                body = re.sub(r"\s+", " ", body_of(path)).strip()
                docs.append(
                    {
                        "tier": tier,
                        "document_id": path.name,
                        "body": body,
                        "loose": loose(body),
                        "chars": len(body),
                    }
                )
    return docs


def find(value: str, docs: list[dict]) -> list[dict]:
    """기대값을 **본문에** 담은 문서를 모두 찾는다."""
    needle = loose(value)
    if not needle:
        return []
    found = []
    for doc in docs:
        at = doc["loose"].find(needle)
        if at < 0:
            continue
        # 느슨한 좌표를 원문 좌표로 되돌릴 수는 없으니, 원문에서 다시 찾는다.
        # 공백이 끼어 못 찾으면 느슨한 비율로 근사 위치를 적는다.
        exact = doc["body"].find(value)
        position = exact if exact >= 0 else int(at / max(1, len(doc["loose"])) * doc["chars"])
        start = max(0, position - QUOTE_CHARS // 3)
        found.append(
            {
                "tier": doc["tier"],
                "document_id": doc["document_id"],
                "position": position,
                "doc_chars": doc["chars"],
                "ratio": round(position / max(1, doc["chars"]), 3),
                "quote": doc["body"][start : start + QUOTE_CHARS],
            }
        )
    return found


def narrow(hits: list[dict], values: list[str], tiers: list[str]) -> list[dict]:
    """후보를 기계적으로 좁힌다. 사람이 고를 일을 줄이는 것이 목적이다.

    짧은 기대 문자열(`0.823`, `720`, `Faithfulness`)은 여러 문서에 나타난다.
    처음 돌렸을 때 연쇄 30문항 중 28개가 여러 문서에 걸렸다. 그대로 두면
    사람이 전부 손으로 골라야 하고, 새 자료가 들어오면 다시 해야 한다.

    두 가지로 좁힌다. 둘 다 데이터에 이미 적혀 있는 정보다.

    1. **기대 계층 안에서만 본다.** 데이터셋이 `expect_tiers`를 적어 두었다.
       "그 벤치마크를 보고한 다음 주 세미나" 문항은 stm이므로 같은 값이
       들어 있는 계획서(ltm)는 후보가 아니다.
    2. **기대값을 더 많이 담은 문서를 고른다.** 기대값이 여럿인 문항은
       그것을 **모두** 담은 문서가 정답일 가능성이 높다. 가장 많이 담은
       문서들만 남긴다.

    그래도 여럿이면 사람이 고른다 — 그때는 진짜로 애매한 문항이다.
    """
    # 같은 문서가 여러 기대값에 걸리면 한 번만 센다. 몇 개를 담았는지 세어 둔다.
    by_doc: dict[str, dict] = {}
    for hit in hits:
        entry = by_doc.setdefault(hit["document_id"], {**hit, "matched_values": []})
        value = hit.get("value")
        if value and value not in entry["matched_values"]:
            entry["matched_values"].append(value)
        # 앞쪽에 나오는 자리를 대표로 남긴다.
        if hit["position"] < entry["position"]:
            entry["position"] = hit["position"]
            entry["quote"] = hit["quote"]
            entry["ratio"] = hit["ratio"]
    picked = list(by_doc.values())

    if tiers:
        in_tier = [h for h in picked if h["tier"] in tiers]
        if in_tier:
            picked = in_tier

    if len(picked) > 1 and len(values) > 1:
        best = max(len(h["matched_values"]) for h in picked)
        if best > 1:
            picked = [h for h in picked if len(h["matched_values"]) == best]
    return picked


def rows_of(name: str) -> list[dict]:
    path = DATA / name
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def expectations(row: dict) -> list[str]:
    """데이터셋마다 기대값 필드 이름이 다르다.

    연쇄는 `expect`, 답변 채점은 `needles`, 라우팅은 `expect_doc`(문서 이름
    조각)을 쓴다. 앞 둘은 본문에서 찾을 문자열이고 라우팅은 이미 문서
    수준이라 다르게 다룬다(`doc_hint`).
    """
    for key in ("expect", "expect_values", "needles"):
        value = row.get(key)
        if isinstance(value, list) and value:
            return [str(v) for v in value]
        if isinstance(value, str) and value:
            return [value]
    return []


def doc_hint(row: dict) -> str:
    """라우팅의 `expect_doc`. 문서 이름 조각이다."""
    return str(row.get("expect_doc") or "").strip()


def match_documents(hint: str, docs: list[dict], tiers: list[str]) -> list[dict]:
    """이름 조각으로 문서를 찾는다. 기대 계층 안에서만 본다.

    라우팅 라벨은 "연구개발계획서"처럼 조각이라 여러 건에 걸린다. 그것을
    그대로 남긴다 — 라우팅 채점은 "기대 문서 **중 하나라도** 후보에
    들어왔나"를 묻기 때문에 목록이 맞다.
    """
    needle = loose(hint)
    if not needle:
        return []
    picked = []
    for doc in docs:
        if tiers and doc["tier"] not in tiers:
            continue
        if needle in loose(doc["document_id"]):
            picked.append(
                {
                    "tier": doc["tier"],
                    "document_id": doc["document_id"],
                    "position": 0,
                    "doc_chars": doc["chars"],
                    "ratio": 0.0,
                    "quote": "",
                    "matched_by": "document_name",
                }
            )
    return picked


def label_file(name: str, docs: list[dict], write: bool) -> None:
    rows = rows_of(name)
    if not rows:
        print(f"{name}: 없음 — 건너뛴다")
        return
    labeled, missing, ambiguous = [], [], []
    for row in rows:
        values = expectations(row)
        hits: list[dict] = []
        for value in values:
            hits.extend({**h, "value": value} for h in find(value, docs))
        if not hits:
            # 라우팅처럼 기대값이 문서 이름 조각인 데이터셋이다.
            hits = match_documents(
                doc_hint(row), docs, list(row.get("expect_tiers") or [])
            )
            if hits:
                values = values or [doc_hint(row)]
        picked = narrow(hits, values, list(row.get("expect_tiers") or []))
        entry = {
            "qid": row.get("qid", ""),
            "question": row.get("question", ""),
            "expect_values": values,
            "expect_documents": picked,
        }
        labeled.append(entry)
        if not picked:
            missing.append(entry)
        elif len(picked) > 1:
            ambiguous.append(entry)

    print(f"\n== {name}  문항 {len(rows)}개")
    print(f"   본문에서 기대값을 찾은 문항   {len(rows) - len(missing)}/{len(rows)}")
    print(f"   문서가 하나로 정해진 문항     {len(rows) - len(missing) - len(ambiguous)}")
    print(f"   여러 문서에 걸린 문항         {len(ambiguous)}  ← 사람이 고를 대상")
    print(f"   본문에 아예 없는 문항         {len(missing)}  ← 지금 채점이 샜던 곳")

    if missing:
        print("\n   [본문에 없음] 기대값이 파일명·머리글에만 있거나 표현이 다르다")
        for entry in missing:
            print(f"     {entry['expect_values']}  |  {entry['question'][:52]}")

    if ambiguous:
        print("\n   [여러 문서] 어느 것이 정답인지 사람이 골라야 한다")
        for entry in ambiguous[:8]:
            print(f"     {entry['question'][:52]}")
            for hit in entry["expect_documents"]:
                print(
                    f"        [{hit['tier']}] {hit['document_id'][:50]}"
                    f"  {hit['position']}자 지점 (문서의 {hit['ratio']*100:.0f}%)"
                )

    # 잘림 손해를 미리 본다 — 청크 전환의 효과를 잴 기준이 여기서 나온다.
    deep = [
        hit
        for entry in labeled
        for hit in entry["expect_documents"]
        if hit["position"] > 6000
    ]
    if deep:
        print(f"\n   [발췌 6,000자 밖에 답이 있는 자리 {len(deep)}건]")
        print("   지금 구조에서는 그 문서가 1위가 되어도 카드에 답이 실리지 않는다.")
        for hit in sorted(deep, key=lambda h: -h["position"])[:6]:
            print(
                f"     {hit['position']:7d}자 지점 (문서의 {hit['ratio']*100:3.0f}%)"
                f"  {hit['document_id'][:48]}"
            )

    if write:
        LABELS.mkdir(parents=True, exist_ok=True)
        out = LABELS / name
        out.write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in labeled) + "\n",
            encoding="utf-8",
        )
        print(f"\n   -> {out.relative_to(ROOT)}")


#: 기본으로 라벨을 붙이는 평가셋.
DEFAULT_FILES = ("routing_cases.jsonl", "session_chains.jsonl", "answer_checks.jsonl")


def main() -> int:
    global SEED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="labels/*.jsonl 생성")
    parser.add_argument(
        "--seed",
        default=None,
        help="시드 루트 폴더 이름. 기본은 memory_seed_prentice",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=None,
        help="라벨을 붙일 평가셋 파일. 여러 번 줄 수 있다",
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.seed:
        SEED = ROOT / args.seed
        if not SEED.is_dir():
            print(f"시드 루트가 없다: {SEED}")
            return 1

    docs = load_documents()
    print(f"시드 루트 {SEED.name}  문서 {len(docs)}건 (본문만, 머리글 제외)")
    for name in args.file or DEFAULT_FILES:
        label_file(name, docs, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
