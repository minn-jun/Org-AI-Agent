"""내보낸 Codex 대화를 MTM 메모리 시드로 바꾼다.

## 왜 MTM인가

`BRANCH_CRITERIA`가 계층을 성격으로 나눈다. STM은 "세미나나 킥오프
회의에서 **구두로 받은 피드백**", MTM은 "주차별 회의자료에 정리한
**기술 조사와 작업 기록**", LTM은 "과제 **행정 문서**"다. 날짜가 아니라
성격이 기준이다 — 실제로 기존 시드도 STM과 MTM이 같은 날짜를 공유한다.

Codex 대화는 작업 기록이다. 무엇을 조사했고 무엇을 바꿨고 어떤 수치가
나왔는지가 들어 있다. 그래서 전부 MTM으로 둔다.

`최근 피드백이야` 스레드는 제목만 보면 STM처럼 보이지만 아니다. 그
스레드의 첫 메시지가 참조하는 `2026-07-08_세미나 피드백 등.txt`는 이미
STM 시드(`2026-07-08-세미나-피드백.json`)로 들어가 있고, 스레드 본문은
그 피드백을 **가지고 한 작업**이다. 피드백 자체가 아니라 반영 기록이다.

## 날짜별로 쪼갠다

가장 긴 스레드는 11일에 걸쳐 187,091자다. 한 문서로 두면 두 가지가
깨진다. "8월 말에 뭘 했지" 같은 날짜 질문의 정답 범위가 11일로 뭉개지고,
BM25 길이 정규화(`b=1.0`)에서 긴 문서가 구조적으로 불리해진다. 뒷부분이
발췌 상한 밖으로 밀리는 문제는 이미 LTM 계획서로 측정하고 있으므로
작업 기록에서 같은 문제를 또 만들 이유가 없다.

그래서 **스레드 × 날짜**가 문서 하나다. 주차별 회의자료 시드와 같은 알이다.

## 무관한 대화를 버리지 않는다

내보낸 스레드의 절반 가까이가 과제와 무관하다(강의 과제 번역, C언어
중간고사, 게임 파티 구성). 버리지 않고 `source_type: codex_offtopic`,
`project: 기타`로 같은 저장소에 넣는다. 지금 평가셋은 정답이 있는 문서만
들어 있어서 **엉뚱한 것을 가져오지 않는가를 재지 못한다.** 이 문서들이
그 역할을 한다.

## 쓰는 법

    python scripts/build_codex_seeds.py            # 미리보기만
    python scripts/build_codex_seeds.py --write

과제명·기관명으로도 관련도를 가리려면 환경변수로 넘긴다. 코드에 적지
않는 이유는 이 저장소가 공개라서다.

    PRENTICE_PROJECT_KEYWORDS="과제명,기관명" python scripts/build_codex_seeds.py

새 로그를 받으면 `--export`에 폴더를 추가해 같은 명령을 다시 돌린다.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: 내보내기 폴더는 저장소 밖에 둔다(실명·로컬 경로가 들어 있다).
EXPORT_BASE = ROOT.parent
DEFAULT_EXPORTS = [
    "Codex_대화_내보내기",
    "Codex_대화_현재PC_DESKTOP-OPF4TTA_2026-10-20",
]

#: 이 글자 수를 못 넘기는 하루치는 시드로 만들지 않는다. "고마워" 한 줄짜리
#: 문서가 후보 카드 한 자리를 차지하면 측정만 흐려진다.
MIN_CHARS = 400

#: 과제 관련도. 이 수를 넘으면 작업 기록, 아니면 방해 문서로 본다.
RELEVANT_HITS = 10

#: 과제명·기관명은 코드에 적지 않고 환경변수로 받는다(`build_prentice_seed.py`와
#: 같은 규칙). 비어 있어도 아래 일반 낱말만으로 관련도가 가려진다.
_PROJECT_WORDS = [
    word.strip()
    for word in os.environ.get("PRENTICE_PROJECT_KEYWORDS", "").split(",")
    if word.strip()
]
KEYWORDS = re.compile(
    "|".join(
        [re.escape(word) for word in _PROJECT_WORDS]
        + [
            r"org_agent",
            r"org_mem",
            r"중기청",
            r"조직지식",
            r"인수인계",
            r"llm_study",
            r"세미나",
            r"RAGAS",
            r"[Jj][Ee][Vv]",
            r"STM",
            r"MTM",
            r"LTM",
            r"프리페치",
            r"라우팅",
            r"에이전트",
            r"메모리",
        ]
    )
)

#: 개인정보가 본문인 문서는 시드에 넣지 않는다. 지나가듯 한 번 언급한
#: 것까지 버리면 작업 기록이 비므로 횟수로 가린다.
PERSONAL = re.compile(r"급여대장|명세서|이체확인증|연봉|인건비|급여")
PERSONAL_LIMIT = 3

#: 남겨도 쓸모가 없고 지우면 깔끔한 것들만 가린다. 실명은 가리지 않는다 —
#: 기존 시드도 파일명에 실명을 쓰고 있고 시드 폴더는 gitignore 대상이다.
MASKS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-or-v1-[A-Za-z0-9_-]{32,}"), "sk-or-v1-REDACTED"),
    (re.compile(r"sk-ant-[A-Za-z0-9_-]{32,}"), "sk-ant-REDACTED"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<메일주소>"),
    (re.compile(r"[Cc]:[/\\]Users[/\\][A-Za-z0-9_]+"), "~"),
]


def mask(text: str) -> str:
    for pattern, replacement in MASKS:
        text = pattern.sub(replacement, text)
    return text


#: 제목을 바로잡는 이름표.
#:
#: 내보내기는 제목이 없는 스레드의 **첫 메시지를 그대로 제목으로** 썼다.
#: 그대로 두면 파일명이 "지금-openspec을-배워서-바이브코딩을-활용해서-나에게"
#: 처럼 된다. 문서 이름은 검색 대상이고 채점에서 `expect_doc`으로 맞춰 보는
#: 값이라 알아볼 수 있어야 한다. 아래 조각이 제목에 들어 있으면 그 이름을 쓴다.
TOPICS: list[tuple[str, str]] = [
    # 과제 관련
    ("오픈라우터에서 https://openrouter.ai/nvidia/nemotron", "헤르메스-Nemotron-실행실험"),
    ("이전 대화에서는 hermes-agent 위주로", "openclaw-설치-실험"),
    ("헤르메스 코드에서 실제로 가져와서 작게", "헤르메스-축소판-구현"),
    ("260618-0701 안에 open-source-agents-latest", "오픈소스-에이전트-문서-정리"),
    ("클로드가 org", "org_agent_mvp-인수인계-검토"),
    ("프로젝트 한번 전부 읽어보고 어느부분이 문제인지", "프로젝트-점검-개선안"),
    ("이번 6월 17일 세미나에서", "6월-17일-세미나-보완"),
    ("최근 피드백이야", "컨텍스트초기화-세미나-피드백-반영"),
    ("bm25, bm25+, 리랭킹", "검색-용어-정리"),
    ("코드 분석 한번 해줘", "코드-분석-깃-연결-확인"),
    ("jev가 어떤 모델이야", "jev-모델-소개"),
    ("하 씨 중기청과제는", "과제-분담-상의"),
    ("지금 openspec을 배워서", "OpenSpec-자유주제-개발"),
    # 과제와 무관한 것(방해 문서)
    ("reinforcement_learning 디렉토리", "강화학습-강의자료-정리"),
    ("이전에 이 pc에 클로드코드를 설치", "클로드코드-설치-확인"),
    ("지능형 정보보호 특론 안에 docx", "정보보호특론-자료-번역"),
    ("지능형정보보호특론에 assignment", "정보보호특론-과제2-작성"),
    ("지능형 정보보호 특론 안에 PEC01", "정보보호특론-PEC02-번역"),
    ("지능형 정보보호 특론에 pet short paper", "정보보호특론-PET-과제-번역"),
    ("ipynb 파일 두개 있는데", "ipynb-템플릿-비교"),
    ("과제/검토 안에 zip파일들", "제출과제-zip-검토"),
    ("그 학술대회에 제출할 논문을 작성", "학술대회-단편논문-작성"),
    ("중간고사/문제 안에 보면 C언어", "C언어-중간고사-문제-검토"),
    ("#include <stdio.h>", "C언어-코드-질문"),
    ("스마트서비스응용 안에 클로드가 자료조사", "스마트서비스응용-자료-검증"),
    ("스마트서비스응용 폴더에서 작업하자", "스마트서비스응용-작업-시작"),
    ("깃 레포지토리 하나를 프라이빗으로", "비공개-레포-공유-방법"),
    ("혹시 메일보내달라하면", "메일-발송-가능-확인"),
    ("이 컴퓨터에서 코덱스로 대화한 로그", "이-PC-코덱스-로그-확인"),
    ("번역해줘", "문서-번역"),
]

#: 제목 끝에 붙는 말버릇. 이름에 남길 것이 아니다.
TAIL = re.compile(
    r"(?:해\s*줄\s*수?\s*있(?:어|나|을까)|해\s*줄래|해\s*줘|줄\s*수?\s*있(?:어|나)"
    r"|있거든|있는데|할\s*수\s*있(?:어|나)|가능(?:해|한지)|부탁해|해\s*봐|해\s*보자"
    r"|어떻게\s*해야해|이거|한번|좀)\s*[?.!]*$"
)
#: 제목 앞에 붙는 군더더기.
HEAD = re.compile(r"^(?:지금|이번|혹시|일단|그래서|근데|하\s*씨|좀|그)\s+")

#: 어디에 있어도 뜻을 보태지 않는 낱말. 토큰이 정확히 이것이면 뺀다.
FILLER = {"이거", "이걸", "한번", "좀", "혹시", "일단", "그냥", "다시", "너가", "니가"}


def topic(title: str) -> str:
    """파일명에 쓸 이름. 기존 시드처럼 한글을 살리고 하이픈으로 잇는다."""
    raw = title.strip()
    for needle, name in TOPICS:
        if needle in raw:
            return name
    # 제목에 첫 메시지가 들어간 스레드다. 첫 줄만 쓰고 군더더기를 떼어 낸다.
    head = raw.splitlines()[0] if raw else "무제"
    head = re.sub(r"^#+\s*", "", head)
    head = re.sub(r"https?://\S+", " ", head)
    head = re.sub(r"[A-Za-z]:[/\\]\S+", " ", head)
    head = re.sub(r"\s+", " ", head).strip()
    for _ in range(3):
        trimmed = TAIL.sub("", HEAD.sub("", head)).strip(" ,.?!")
        if trimmed == head:
            break
        head = trimmed
    head = " ".join(word for word in head.split() if word not in FILLER)
    # 밑줄은 살린다 — `llm_study`, `org_agent_mvp`가 그대로 검색에 걸려야 한다.
    head = re.sub(r"[^0-9A-Za-z가-힣_]+", "-", head).strip("-")
    if len(head) > 40:
        head = head[:40].rsplit("-", 1)[0] or head[:40]
    return (head or "무제").strip("-")


def usable(message: dict) -> bool:
    """화면에 나간 것만 남긴다.

    `phase`가 `commentary`인 답변은 "읽고 있습니다" 같은 진행 안내다.
    전체 메시지의 2/3가 여기에 해당한다. 근거로 들어가면 안 된다.
    """
    if message.get("role") == "user":
        return True
    return message.get("phase") == "final_answer"


def day_of(message: dict) -> str:
    stamp = message.get("created_at_ms") or 0
    return datetime.datetime.fromtimestamp(stamp / 1000).strftime("%Y-%m-%d")


def body_of(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        text = mask((message.get("text") or "").strip())
        if not text:
            continue
        head = "## 요청" if message.get("role") == "user" else "## 답변"
        parts.append(f"{head}\n\n{text}")
    return "\n\n".join(parts)


def summary_of(body: str, limit: int = 220) -> str:
    """기존 시드와 같은 방식 — 본문 앞부분을 한 줄로 눌러 담는다."""
    flat = re.sub(r"\s+", " ", re.sub(r"^##.*$", "", body, flags=re.M)).strip()
    return flat[:limit]


def document(
    *,
    date: str,
    title: str,
    body: str,
    origin: str,
    relevant: bool,
) -> str:
    header = {
        "memory_tier": "mtm",
        "source_type": "codex_worklog" if relevant else "codex_offtopic",
        "project": "진행 과제" if relevant else "기타",
        "title": title,
        "date": date,
        "permission_scope": "project_team",
        "status": "active",
        "summary": summary_of(body),
        "summary_source": "auto",
        "origin": origin,
    }
    lines = ["---"]
    for key, value in header.items():
        lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines) + body + "\n"


def threads_of(folder: Path) -> list[dict]:
    data = json.loads((folder / "대화_데이터.json").read_text(encoding="utf-8"))
    return list(data.get("threads") or [])


def build(exports: list[str], out: Path, write: bool) -> int:
    rows: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for export in exports:
        folder = EXPORT_BASE / export
        if not folder.is_dir():
            print(f"내보내기 폴더가 없다 — 건너뛴다: {folder}")
            continue
        for thread in threads_of(folder):
            title = str(thread.get("title") or "")
            if thread.get("category") == "internal" or title.startswith("[내부 보조기록]"):
                skip("내부 보조기록")
                continue
            messages = [m for m in (thread.get("messages") or []) if usable(m)]
            if not messages:
                skip("원본 로그 없음")
                continue
            blob = " ".join((m.get("text") or "") for m in messages)
            relevant = len(KEYWORDS.findall(blob)) >= RELEVANT_HITS

            by_day: dict[str, list[dict]] = {}
            for message in messages:
                by_day.setdefault(day_of(message), []).append(message)

            for date, day_messages in sorted(by_day.items()):
                body = body_of(day_messages)
                if len(body) < MIN_CHARS:
                    skip(f"{MIN_CHARS}자 미만")
                    continue
                if len(PERSONAL.findall(body)) >= PERSONAL_LIMIT:
                    skip("개인정보 본문")
                    continue
                piece = topic(title)
                rows.append(
                    {
                        "date": date,
                        "slug": piece,
                        "title": f"{date} Codex 작업기록 {piece.replace('-', ' ')}",
                        "body": body,
                        "origin": f"Codex 대화 내보내기 / {export} / {title.splitlines()[0][:60]}",
                        "relevant": relevant,
                        "turns": len(day_messages),
                    }
                )

    # 같은 날 같은 제목이 겹치면 번호를 붙인다.
    seen: dict[str, int] = {}
    for row in rows:
        stem = f"{row['date']}-codex-{row['slug']}"
        seen[stem] = seen.get(stem, 0) + 1
        row["name"] = stem if seen[stem] == 1 else f"{stem}-{seen[stem]}"

    target = out / "mtm"
    if write:
        target.mkdir(parents=True, exist_ok=True)
        for row in rows:
            (target / f"{row['name']}.md").write_text(
                document(
                    date=row["date"],
                    title=row["title"],
                    body=row["body"],
                    origin=row["origin"],
                    relevant=row["relevant"],
                ),
                encoding="utf-8",
            )

    work = [r for r in rows if r["relevant"]]
    noise = [r for r in rows if not r["relevant"]]

    def chars(items: list[dict]) -> int:
        return sum(len(item["body"]) for item in items)

    print(f"{'썼다' if write else '(미리보기)'}  {target}")
    print(f"  작업 기록  {len(work):3d}개  {chars(work):9,}자")
    print(f"  방해 문서  {len(noise):3d}개  {chars(noise):9,}자")
    print(f"  합계      {len(rows):3d}개  {chars(rows):9,}자")
    if rows:
        longest = max(rows, key=lambda r: len(r["body"]))
        print(f"  가장 긴 문서  {len(longest['body']):,}자  {longest['name']}")
        print(
            f"  날짜 범위     {min(r['date'] for r in rows)}"
            f" ~ {max(r['date'] for r in rows)}"
        )
    if skipped:
        print("  건너뛴 것: " + ", ".join(f"{k} {v}" for k, v in sorted(skipped.items())))
    return 0


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="append", default=None)
    parser.add_argument("--out", default="memory_seed_codex")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    return build(args.export or DEFAULT_EXPORTS, ROOT / args.out, args.write)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
