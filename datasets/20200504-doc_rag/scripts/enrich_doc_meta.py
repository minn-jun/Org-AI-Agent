"""20200504 과제 코퍼스 전용 전처리 — 문서 메타데이터를 채운다.

검색기(`org_agent_mvp/ltm_corpus.py`)는 특정 코퍼스의 폴더 이름이나 파일명 습관을 모른다.
이 과제 폴더에서만 통하는 규칙을 여기로 옮겨, 결과를 `document_meta.jsonl`로 떨군다.
(2026-09-15 분리. 규칙과 수치 근거는 원래 ltm_corpus.py에 있던 주석 그대로다.)

    python enrich_doc_meta.py              # export/.../chunks.jsonl -> 같은 폴더 document_meta.jsonl

출력 한 줄 = 한 문서
    {"doc_id", "doc_type", "version_group", "version_rank": [최종, 주, 부], "is_final", "project"}

과제명·기관명은 `corpus_profile.json`(이 폴더, git 밖)에서 읽는다.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CORPUS = DATA / "export" / "knowledge_service_core_tech_20200504" / "chunks.jsonl"
PROFILE_PATH = DATA / "corpus_profile.json"
_DEFAULT_PROFILE = {"project": "프로젝트", "organizations": []}


def load_profile(path: Path | None = None) -> dict:
    target = Path(path) if path else PROFILE_PATH
    if not target.exists():
        return dict(_DEFAULT_PROFILE)
    return {**_DEFAULT_PROFILE, **json.loads(target.read_text(encoding="utf-8"))}


_PROFILE = load_profile()

#: 이 코퍼스가 통째로 속한 과제. STM/MTM의 project 값과 같아야 필터가 맞는다.
CORPUS_PROJECT: str = _PROFILE["project"]

#: 참여 기관 표기. 목록은 프로파일에서 온다.
_ORGANIZATIONS: list[str] = list(_PROFILE["organizations"])


def _org_pattern(suffix_only: bool) -> "re.Pattern[str]":
    """기관 표기 정규식. 목록이 비면 아무것도 매칭하지 않는다."""
    if not _ORGANIZATIONS:
        return re.compile(r"(?!)")
    body = "|".join(re.escape(o) for o in _ORGANIZATIONS)
    if suffix_only:
        return re.compile(r"(?i)[\s_\-]+(?:" + body + r")\s*$")
    return re.compile(r"(?i)(?:" + body + r")")

#: 폴더 이름에서 문서 성격을 뽑는다. 파일 형식(pdf/hwp)을 source_type으로 쓰면
#: analyzer의 enum이 형식 이름으로 오염돼서, STM/MTM과 같은 의미 공간으로 옮긴다.
#: 앞에 있는 규칙이 먼저 맞는다.
_TYPE_RULES: list[tuple[str, str]] = [
    ("협약변경", "agreement"),
    ("협약", "agreement"),
    ("최종보고서", "official_report"),
    ("차년도 보고서", "official_report"),
    ("단계 평가 결과", "evaluation_result"),
    ("종합의견서", "evaluation_result"),
    ("대면평가", "evaluation_result"),
    ("계획서 작업", "plan_document"),
    ("개발 계획", "plan_document"),
    ("특허", "patent_document"),
    ("학술대회", "publication"),
    ("시험평가", "test_report"),
    ("시범 서비스", "commercialization"),
    ("MOU", "commercialization"),
    ("정산", "budget_document"),
    ("예산", "budget_document"),
    ("연구비", "budget_document"),
    ("현물부담", "budget_document"),
    ("퇴직급여", "budget_document"),
    ("외부연구원", "personnel_document"),
    ("전문가활용", "personnel_document"),
    ("화면 설계", "design_document"),
    ("디자인 컨설팅", "design_document"),
    ("HW 설계", "design_document"),
    ("회의", "meeting_document"),
    ("공고", "reference_material"),
    ("사전 준비", "reference_material"),
]


# ------------------------------------------------------------------ 버전 접기
#
# 조직 문서에는 같은 문서의 작업본이 잔뜩 쌓인다. 이 코퍼스는 청크의 25%가
# 글자까지 똑같은 중복이고, 현실 질의 12개의 상위 5건 중 31.7%가
# 같은 계열의 반복이었다("차단계 사업계획서_v3 / _v4 / _v8 / _v9 / _v10").
#
# 인덱스에서 지우지는 않는다. 연구 단계에서는 버전 이력 자체가 자료다.
# 검색 결과에서만 한 건으로 접고, 나머지는 몇 건인지만 알려준다.

#: 계열 키가 이보다 짧아지면 더 깎지 않는다.
#: 제목이 통째로 사라져 관련 없는 문서가 한 덩어리가 되는 것을 막는다
#: (실측: 이 가드가 없을 때 Article... / BOM... / Scan... 10건이 빈 키로 합쳐졌다).
_MIN_FAMILY_KEY = 8

_VERSION_TAG = re.compile(
    r"""(?ix)
    [\s_\-]+(?:
      # 작업자 이니셜은 버전 번호에 붙어 있을 때만 뗀다.
      # 단독으로 떼면 기관 약어(_ABC)를 이니셜(_jic)로 오인한다 —
      # 기관 표기는 버전이 아니라 별개 문서를 가리키므로 남겨야 한다.
      [A-Za-z]{2,4}\d?[\s_\-]+v\d+(?:[._-]\d+)*       # _aej_v3
    | v\d+(?:[._-]\d+)*[\s_\-]+[A-Za-z]{2,4}\d?       # _v16_jic
    | v\d+(?:[._-]\d+)*                               # _v1 _v10 _v1.1
    | \d+차(?:\s*수정)?
    | fina?l | 최종본? | 완성본?
    | 수정(?:본|안|판)? | 재수정
    | 취합본? | 통합본? | 정리본?
    | 원본 | 그림원본 | 그림작업
    | 배포용 | 제출용 | 인쇄용
    | copy | 사본
    # 검토 상태 표기. 앞에 사람이 붙는다 — "_○○○ 검토", "_교수님 검토", "_검토전"
    | [가-힣]{0,4}\s*검토(?:전|후|본|중)?
    | [가-힣]{0,4}\s*(?:수정본|확인본|반영본|보완본|정리본)
    | \d{1,2}                                          # _01 _02
    )\s*$
  | \s*\(\d+\)\s*$
    """
)


#: 기관 표기. 버전 꼬리표처럼 생겼지만 **다른 문서**를 가리킨다.
#:
#: "확인서_A기관"와 "확인서_B기관"는 같은 서식을 두 기관이 각각 작성한 별개 서류다.
#: 대표 서명도 내용도 달라서 접으면 한쪽 기관 서류가 통째로 숨는다.
#: 실측으로 기관 표기가 붙은 문서가 91건, 그중 쌍을 이루는 계열이 8개다.
_ORG_TAG = _org_pattern(suffix_only=False)

#: 제목 끝에 붙은 기관 표기. 버전 번호가 있는 제목에서만 뗀다.
_ORG_SUFFIX = _org_pattern(suffix_only=True)

#: 사람 이름 프리픽스가 없는 엄격한 버전. 기관 표기를 지켜야 할 때 쓴다.
_VERSION_TAG_STRICT = re.compile(
    r"""(?ix)
    [\s_\-]+(?:
      [A-Za-z]{2,4}\d?[\s_\-]+v\d+(?:[._-]\d+)*
    | v\d+(?:[._-]\d+)*[\s_\-]+[A-Za-z]{2,4}\d?
    | v\d+(?:[._-]\d+)*
    | \d+차(?:\s*수정)?
    | fina?l | 최종본? | 완성본?
    | 수정(?:본|안|판)? | 재수정
    | 취합본? | 통합본? | 정리본?
    | 원본 | 그림원본 | 그림작업
    | 배포용 | 제출용 | 인쇄용
    | copy | 사본
    | 검토(?:전|후|본|중)?
    | \d{1,2}
    )\s*$
  | \s*\(\d+\)\s*$
    """
)


def family_key(title: str) -> str:
    """버전 꼬리표를 떼어 같은 문서 계열을 하나의 키로 만든다.

    `_○○○ 검토`처럼 사람 이름이 앞에 붙는 꼬리표가 있어서 한글 프리픽스를
    허용하는데, 그대로 두면 `_A기관 정리본`의 기관명까지 먹는다.
    그래서 떼어낼 구간에 기관 표기가 섞이면 프리픽스 없는 규칙으로 다시 잡는다.

        보고서_A기관 정리본  ->  보고서_A기관     (기관은 남는다)
        보고서_A기관 최종본  ->  보고서_A기관
        보고서_v9_○○○ 검토 ->  보고서            (사람 이름은 떨어진다)
    """
    # 버전 번호가 붙은 제목에서는 기관 표기가 "누가 고쳤나"를 뜻한다.
    #   발표자료_v7.2_A기관 수정본   -> A기관가 고친 v7.2. 기관은 문서 구분이 아니다
    #   확인서_A기관               -> A기관의 확인서. 기관이 문서를 가른다
    # 그래서 버전 번호가 없을 때만 기관 표기를 지킨다.
    protect_org = not _VERSION_NUM.search(title)
    for _ in range(8):
        m = _VERSION_TAG.search(title)
        if not m:
            if protect_org:
                break
            m = _ORG_SUFFIX.search(title)   # 기관만 남았으면 그것도 뗀다
            if not m:
                break
        elif protect_org and _ORG_TAG.search(title[m.start():]):
            m = _VERSION_TAG_STRICT.search(title)
            if m is None or _ORG_TAG.search(title[m.start():]):
                break
        candidate = title[: m.start()].strip(" _-")
        if len(candidate) < _MIN_FAMILY_KEY:
            break
        title = candidate
    return re.sub(r"\s+", " ", title).strip()


_VERSION_NUM = re.compile(r"(?i)(?:^|[\s_\-])v(\d+)(?:[._-](\d+))?(?![\d])")
_FINAL_TAG = re.compile(r"(?i)fina?l|최종")


def version_rank(title: str) -> tuple[int, int, int]:
    """계열 안에서 어느 문서가 최신인지 재는 값. 클수록 최신이다.

    파일 수정일은 못 쓴다 — 이 코퍼스는 통째로 내려받은 사본이라
    698건이 전부 같은 날짜(2026-08-26)로 찍혀 있다. 그래서 제목의 버전 표기를 본다.

        _v17            -> (0, 17, 0)
        _v7.2_A기관 수정본 -> (0, 7, 2)
        _v01_jic        -> (0, 1, 0)
        _Final_..._v2   -> (1, 2, 0)
        _최종(그림원본)     -> (1, 0, 0)
        (꼬리표 없음)      -> (0, 0, 0)

    `최종`/`Final`을 버전 번호보다 앞에 두는 이유는, 한국어 문서 관행에서
    `_최종`이 붙으면 번호가 매겨진 작업본을 대체한다고 보기 때문이다.
    """
    major = minor = 0
    for m in _VERSION_NUM.finditer(title):
        cand = (int(m.group(1)), int(m.group(2) or 0))
        if cand > (major, minor):
            major, minor = cand
    return (1 if _FINAL_TAG.search(title) else 0, major, minor)


#: 실제로 제출된 산출물이 놓인 폴더. 파일명 버전보다 강한 신호다.
#:
#: 실측: 계열 안에 이 폴더의 문서가 있을 때, 그게 버전 최고와 일치한 비율이 89%다.
#: 게다가 버전 번호가 아예 없는 최종본을 잡아낸다 — 실제로
#: "신청용 계획서(PART II)" 계열은 버전 최고가 `_v4`(작업 폴더)인데
#: 진짜 제출본은 꼬리표 없이 `06_최종제출` 폴더에 있었다.
_FINAL_DIR = re.compile(r"최종\s*제출|최종제출|제출\s*자료|최종\s*전달|업로드본|제출 서류")


def classify(rel_path: str) -> str:
    for keyword, label in _TYPE_RULES:
        if keyword in rel_path:
            return label
    return "official_document"


def document_meta(doc_id: str, title: str, rel_path: str) -> dict:
    return {
        "doc_id": doc_id,
        "doc_type": classify(rel_path),
        "version_group": family_key(title),
        "version_rank": list(version_rank(title)),
        "is_final": bool(_FINAL_DIR.search(rel_path)),
        "project": CORPUS_PROJECT,
    }


def main() -> int:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else CORPUS
    seen: dict[str, dict] = {}
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            doc_id = row["source_id"]
            if doc_id in seen:
                continue
            meta = row["metadata"]
            seen[doc_id] = document_meta(doc_id, meta.get("title") or doc_id, meta.get("source_path", ""))
    out = corpus.with_name("document_meta.jsonl")
    out.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in seen.values()), encoding="utf-8")
    types: dict[str, int] = {}
    for m in seen.values():
        types[m["doc_type"]] = types.get(m["doc_type"], 0) + 1
    print(f"documents {len(seen)} -> {out}")
    print("doc_type", dict(sorted(types.items(), key=lambda kv: -kv[1])))
    print("version groups", len({m["version_group"] for m in seen.values()}), "| is_final", sum(m["is_final"] for m in seen.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
