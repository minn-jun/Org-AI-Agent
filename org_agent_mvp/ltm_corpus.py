"""실제 과제 문서를 LTM으로 붙이는 어댑터.

`datasets/20200504-doc_rag/export/.../chunks.jsonl`(doc_rag가 파싱한 883문서 26,586청크)을 읽어
MemoryStore와 같은 모양의 근거 카드를 돌려준다.

## 왜 파일로 떨구지 않는가

STM/MTM처럼 `ltm/` 폴더에 md 파일로 쓰면 세 가지가 깨진다.

1. **길이 편향** — `MemoryStore._score()`에는 길이 정규화가 없다.
   STM/MTM 문서는 중앙값 436자인데 과제 문서는 중앙값 2,158자, 최대 2,019,319자다.
   긴 문서가 질의어를 여러 번 포함해서 무조건 이긴다.
2. **파일명 충돌** — 근거 id가 `ev_{tier}_{path.stem}`인데,
   883문서 중 79개 이름이 중복된다(총 166건). gold 라벨을 붙일 수 없다.
3. **채점 비용** — 질의마다 문서 전체를 다시 토크나이즈해서 2.4초가 걸린다.

## 어떻게 푸는가

| 문제 | 처리 |
|---|---|
| 길이 편향 | 청크 단위로 채점한다. 청크 평균 754자로 STM/MTM과 자릿수가 같다 |
| 파일명 충돌 | 근거 id를 `ev_ltm_{doc_id}`로 쓴다. doc_id는 상대경로 sha1 앞 16자라 고유하다 |
| 채점 비용 | 로드할 때 역색인을 한 번 만들고, 질의어가 있는 청크만 훑는다 |

검색은 청크 단위, 근거는 문서 단위다.
한 문서의 여러 청크가 걸리면 최고점 청크만 대표로 올리고,
그 청크의 본문을 인용문으로 쓴다. 그래야 STM/MTM과 같은
"문서 하나 = 근거 하나" 계약이 유지된다.

## 최신성

LTM에는 최신성 보정을 걸지 않는다.
파일 수정일은 문서의 시점이 아니라 파일을 마지막에 만진 날이라
"오늘", "최근" 같은 질의에 공식 계획서가 딸려 올라오면 오히려 틀린다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import dense as dense_mod
from .scoring import Bm25Params, freq_weight, scorer_name
from .tokenizer import tokenize, tokenize_many

# --------------------------------------------------------------- 코퍼스 프로파일
#
# 과제명과 참여 기관명은 코드가 아니라 **그 코퍼스의 사실**이다.
# 실제 연구 자료를 다루면 저장소에 올릴 수 없는 값이 되므로 파일로 뺐다.
#
#   config/corpus_profile.json   (gitignore 대상)
#     {
#       "project": "○○ 과제",
#       "organizations": ["기관명", "기관 약어", ...]
#     }
#
# 파일이 없으면 기본값으로 돈다. 기관 목록이 비면 기관 표기 보호가 꺼질 뿐
# 나머지 동작(버전 접기, 보일러플레이트 감쇠)은 그대로다.

_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / "corpus_profile.json"
_DEFAULT_PROFILE = {"project": "프로젝트", "organizations": []}


def load_profile(path: Path | None = None) -> dict:
    target = Path(path) if path else _PROFILE_PATH
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

#: 이 수를 넘는 계열에 같은 본문이 나오면 서식으로 보고 감점한다.
#: 표본 확인 결과 3계열까지는 작업 파일 사이에 복사된 진짜 내용이었고,
#: 4계열부터 계약 문구·서식 지침 같은 것이 나왔다.
_BOILERPLATE_MIN_FAMILIES = 3

#: 토큰화를 몇 건씩 묶어 넘길지. 메모리와 속도의 절충이다.
_TOKENIZE_BATCH = 4000

#: 표의 빈 셀. 채움 정도를 재는 데 쓴다.
_EMPTY_CELL = re.compile(r"\|\s{0,3}(?=\|)")


def _text_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _tokenize(text: str) -> list[str]:
    """memory_store와 같은 토큰화를 쓴다. 한쪽만 바꾸면 매칭이 조용히 깨진다."""
    return tokenize(text)


def classify(rel_path: str) -> str:
    for keyword, label in _TYPE_RULES:
        if keyword in rel_path:
            return label
    return "official_document"


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    rel_path: str
    source_type: str
    stage: str
    modified_at: str
    n_chunks: int
    family: str = ""       # 버전 꼬리표를 뗀 계열 키
    final_dir: int = 0     # 최종 제출 폴더에 있으면 1
    cells: int = 0         # 표 셀 수
    empty_cells: int = 0   # 그중 빈 셀


class LtmCorpus:
    """청크 역색인 + 문서 단위 근거 집계."""

    def __init__(self, path: Path, project: str = CORPUS_PROJECT):
        self.path = Path(path)
        self.project = project
        self.documents: dict[str, CorpusDocument] = {}

        # 청크는 인덱스 정렬 배열로 들고 있는다. 청크마다 dict를 만들면
        # 26,586개 × 토큰 dict라 메모리가 수백 MB로 뛴다.
        self._chunk_doc: list[str] = []      # 청크 -> doc_id
        self._chunk_text: list[str] = []
        self._chunk_title_l: list[str] = []
        self._chunk_hash: list[bytes] = []   # 본문 해시. 버전 간 동일 문단을 접는 데 쓴다
        self._chunk_len: list[int] = []      # 청크 토큰 수. BM25 길이 정규화에 쓴다

        # token -> array('i') [청크번호, 빈도, 청크번호, 빈도, ...]
        self._postings: dict[str, array] = {}

        # 청크 본문이 몇 개의 *계열*에 걸쳐 나오는지. 보일러플레이트 감쇠에 쓴다.
        self._chunk_weight: list[float] = []

        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        postings: dict[str, array] = {}
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                meta = row["metadata"]
                doc_id = row["source_id"]
                if doc_id not in self.documents:
                    self.documents[doc_id] = CorpusDocument(
                        doc_id=doc_id,
                        title=meta.get("title") or doc_id,
                        rel_path=meta.get("source_path", ""),
                        source_type=classify(meta.get("source_path", "")),
                        stage=meta.get("stage", ""),
                        modified_at=str(meta.get("modified_at", ""))[:10],
                        n_chunks=0,
                        family=family_key(meta.get("title") or doc_id),
                        final_dir=1 if _FINAL_DIR.search(meta.get("source_path", "")) else 0,
                    )
                doc = self.documents[doc_id]
                doc.n_chunks += 1
                doc.cells += row["text"].count("|")
                doc.empty_cells += len(_EMPTY_CELL.findall(row["text"]))

                text = row["text"]
                self._chunk_doc.append(doc_id)
                self._chunk_text.append(text)
                self._chunk_title_l.append(doc.title.lower())
                self._chunk_hash.append(_text_hash(text))

        # 토큰화는 한 건씩 부르지 않고 묶어서 넘긴다.
        # 형태소 분석기는 호출 비용이 커서 차이가 크다 — 실측으로 26,031청크에
        # 개별 호출 158초, 배치 호출 14초였다(11.5배).
        for start in range(0, len(self._chunk_text), _TOKENIZE_BATCH):
            stop = start + _TOKENIZE_BATCH
            batch = [
                f"{self.documents[self._chunk_doc[i]].title}\n{self._chunk_text[i]}"
                for i in range(start, min(stop, len(self._chunk_text)))
            ]
            for offset, tokens in enumerate(tokenize_many(batch)):
                idx = start + offset
                counts: dict[str, int] = {}
                for token in tokens:
                    counts[token] = counts.get(token, 0) + 1
                self._chunk_len.append(len(tokens))
                for token, n in counts.items():
                    post = postings.get(token)
                    if post is None:
                        post = postings[token] = array("i")
                    post.append(idx)
                    post.append(n)
        self._postings = postings
        self._compute_boilerplate_weights()
        # BM25는 코퍼스 통계가 필요하다. 청크 수와 평균 길이를 여기서 굳힌다.
        n = len(self._chunk_len) or 1
        self._bm25 = Bm25Params(n_docs=n, avg_len=sum(self._chunk_len) / n)
        # 3단계. 켜져 있을 때만 임베딩 인덱스를 붙인다.
        # 첫 실행은 인코딩이 필요해 오래 걸리고, 이후에는 캐시에서 읽는다.
        self._dense = None
        if dense_mod.enabled():
            self._dense = dense_mod.DenseIndex(self.path, self._dense_texts())

    def _compute_boilerplate_weights(self) -> None:
        """본문이 여러 계열에 걸쳐 나올수록 감점한다.

        서식 목차나 빈 표 헤더 같은 문구는 문서를 가리지 않고 나온다.
        실측으로 청크의 9.4%가 4개 이상 계열에 걸쳐 있었고,
        가장 널리 퍼진 것들은 전부 서식이었다.

            12계열 | 연구시설 ․ 장비명 | 규격 | 구입단가 (천원) | 구입연도 | ...
             9계열 | [ 사업계획서 서식 PART II - 직접 작성하여 전산으로 업로드 ... ] 목 차

        같은 계열 안에서만 반복되는 것(버전 사본)은 감점하지 않는다.
        그건 접기가 처리하고, 내용 자체는 그 문서의 진짜 본문이기 때문이다.

        **3계열까지는 그냥 둔다.** 표본을 열어 보니 성격이 달랐다.

            2계열  "200506_미정리 조사 자료" / "200520_그림자료"
                   -> 시각장애인 인구 전망. 작업 파일 사이에 복사된 진짜 내용이다.
            3계열  "200511_그림 작업" / "200511_그림개요" / "200520_그림자료"
                   -> 같은 도해. 역시 내용이다.
            4계열  "과업지시서_국내 선행특허 조사" / "_비대면 상담" / "_VR 디스플레이"
                   -> 계약 문구. 문서를 가리지 않고 나오는 서식이다.

        서로 다른 4개 이상의 계열에 같은 문단이 나오면 그건 그 문서의 내용이
        아니라 서식이다. 거기서부터 감점한다.

        IDF 꼴(log(1+N/df))도 재봤는데 이 코퍼스에서는 너무 완만했다 —
        566계열 기준으로 12계열짜리 서식조차 0.61밖에 안 깎였다.
        """
        families_of: dict[bytes, set[str]] = defaultdict(set)
        for idx, digest in enumerate(self._chunk_hash):
            doc = self.documents[self._chunk_doc[idx]]
            families_of[digest].add(doc.family or doc.title)
        self._chunk_weight = []
        for digest in self._chunk_hash:
            df = len(families_of[digest])
            self._chunk_weight.append(
                1.0 if df <= _BOILERPLATE_MIN_FAMILIES else 1.0 / (1.0 + math.log(df - 2))
            )

    def _dense_texts(self) -> list[str]:
        """임베딩에 넣을 문자열. sparse 색인과 같은 구성이어야 한다."""
        return [
            self.documents[self._chunk_doc[i]].title + "\n" + self._chunk_text[i]
            for i in range(len(self._chunk_text))
        ]

    # ----------------------------------------------------------------- score

    def search(
        self,
        query_tokens: list[str],
        *,
        query_text: str = "",
        filters: dict[str, Any] | None = None,
        filter_penalty: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[float, dict[str, Any]]]:
        """문서 단위 근거 카드를 점수 높은 순으로 돌려준다.

        `query_text`는 임베딩 검색에만 쓴다. 토큰이 아니라 원문 문장이 필요하다.
        """
        if not query_tokens:
            return []

        # 1) 질의어가 든 청크만 훑는다. 없는 토큰은 posting이 아예 없다.
        #    점수 함수는 scoring 모듈이 정한다(RETRIEVER_SCORER).
        #    BM25는 토큰마다 df가 필요한데, posting 길이가 곧 df라 공짜로 얻는다.
        chunk_score: dict[int, float] = defaultdict(float)
        use_bm25 = scorer_name() == "bm25"
        params = self._bm25 if use_bm25 else None
        # 중복만 없애고 **순서는 유지한다**. set으로 돌리면 안 된다.
        #
        # 파이썬은 프로세스마다 문자열 해시를 다르게 잡아서(PYTHONHASHSEED)
        # set의 순회 순서가 실행마다 바뀐다. 그러면 아래 두 가지가 따라 흔들린다.
        #
        #   - `chunk_score`에 청크가 들어가는 순서. 점수가 같은 청크 중
        #     어느 것이 문서 대표가 되는지가 뒤집히고, 대표가 바뀌면 그 청크의
        #     본문 해시가 달라져서 계열 접기 결과까지 통째로 달라진다.
        #   - 부동소수점 덧셈 순서. 같은 항을 더해도 1e-16 수준의 차이가 난다.
        #
        # 실측으로 같은 명령을 반복했을 때 실코퍼스 60건의 재현율이
        # 62.0%와 62.9% 사이를 오갔고 LTM MRR이 0.358~0.368로 흔들렸다.
        # 측정 도구가 재현되지 않으면 단계 간 비교 자체가 성립하지 않는다.
        #
        # 이 한 줄이면 충분하다. 아래의 `chunk_score` 순회와 정렬은 전부
        # 삽입 순서를 따르므로, 삽입 순서가 결정적이면 나머지도 결정적이다.
        # 동점 처리 규칙을 여기서 더 손대면 재현성과 무관하게 결과가 바뀐다.
        for token in dict.fromkeys(query_tokens):
            post = self._postings.get(token)
            if post is None:
                continue
            df = len(post) // 2
            for i in range(0, len(post), 2):
                idx = post[i]
                tf = post[i + 1]
                if params is not None:
                    w = params.weight(tf, self._chunk_len[idx], df)
                else:
                    w = freq_weight(tf)
                chunk_score[idx] += w * self._chunk_weight[idx]

        # 1-b) 임베딩을 켰으면 sparse 순위와 dense 순위를 RRF로 합친다.
        #      점수를 안 쓰고 순위만 쓰므로 BM25(0~30)와 코사인(0~1)의
        #      눈금 차이가 문제되지 않는다.
        if self._dense is not None and query_text:
            sparse_ranked = sorted(chunk_score, key=chunk_score.get, reverse=True)
            max_sparse = chunk_score[sparse_ranked[0]] if sparse_ranked else 0.0
            sparse_ranked = sparse_ranked[: dense_mod.TOP_N]
            dense_ranked = self._dense.top_n(query_text, n=dense_mod.TOP_N)
            fused = dense_mod.rrf_merge(sparse_ranked, dense_ranked)

            # RRF 점수는 1/(60+순위)라 0.003~0.033 범위다. sparse 점수(수~수십)와
            # 자릿수가 달라서 그대로 쓰면 두 군데가 깨진다.
            #
            #   - 아래에서 더하는 제목 보너스(+2.0)가 점수를 통째로 덮어쓴다
            #   - prefetch의 전역 정규화에서 LTM만 눌려 컷에 걸린다
            #     (실측: LTM 최고점이 23.58 -> 4.03, LTM MRR 0.450 -> 0.125)
            #
            # 그래서 융합은 **순위를 정하는 데만** 쓰고, 눈금은 sparse가 쓰던 것을
            # 그대로 유지한다. 최고점을 맞춰 비례 확대하는 것으로 충분하다.
            # sparse가 한 건도 못 찾으면 max_sparse가 0이라 scale도 0이 된다.
            # 그러면 dense가 찾아 온 청크가 전부 0점이 되어 아래 `final > 0`
            # 필터에서 떨어진다 — dense를 켰는데도 결과가 0건이 된다.
            # 어휘 불일치를 푸는 게 3단계의 목적이므로 그 경우를 살려 둔다.
            #
            # 다만 기준으로 삼을 sparse 점수가 없으므로 눈금을 지어내지 않는다.
            # 최고점을 1.0으로 둔다 — 토큰 하나가 한 번 걸린 약한 일치와 같은
            # 무게다. 다른 계층에 진짜 어휘 일치가 있으면 그쪽이 이기고,
            # 아무것도 없을 때만 표면에 올라온다.
            max_fused = max(fused.values(), default=0.0)
            if max_fused <= 0:
                scale = 1.0
            elif max_sparse > 0:
                scale = max_sparse / max_fused
            else:
                scale = 1.0 / max_fused

            # 보일러플레이트 감쇠는 여기서도 유지한다. 서식 문단은 dense에서도
            # 어느 질의에나 적당히 가까워서 그냥 두면 다시 올라온다.
            chunk_score = defaultdict(
                float,
                {i: s * scale * self._chunk_weight[i] for i, s in fused.items()},
            )

        if not chunk_score:
            return []

        # 2) 제목 가중치는 후보가 정해진 뒤에만 본다.
        #    MemoryStore._score()의 +2.0(제목) / +1.5(과제명)과 같은 값을 쓴다.
        lowered = list(dict.fromkeys(query_tokens))
        project_l = self.project.lower()
        project_bonus = 1.5 * sum(1 for t in lowered if t in project_l)
        for idx, base in list(chunk_score.items()):
            title_l = self._chunk_title_l[idx]
            bonus = 2.0 * sum(1 for t in lowered if t in title_l)
            chunk_score[idx] = base + bonus + project_bonus

        # 3) 문서 단위로 접는다. 최고점 청크가 그 문서를 대표한다.
        best: dict[str, tuple[float, int]] = {}
        for idx, score in chunk_score.items():
            doc_id = self._chunk_doc[idx]
            if doc_id not in best or score > best[doc_id][0]:
                best[doc_id] = (score, idx)

        # 4) 필터는 점수 배수로 적용한다. MemoryStore와 같은 방식이다.
        filters = filters or {}
        scored: list[tuple[float, str, int]] = []
        for doc_id, (score, idx) in best.items():
            weight = self._filter_weight(self.documents[doc_id], filters, filter_penalty)
            final = score * weight
            if final > 0:
                scored.append((final, doc_id, idx))
        # 5) 계열별로 모은다. 계열 키가 같거나 대표 청크 본문이 같으면 한 묶음이다.
        #
        #    묶음의 순위는 그 안에서 가장 높은 점수로 정한다(가장 잘 맞은 근거).
        #    반면 화면에 세울 대표 문서는 **버전이 가장 높은 것**으로 고른다.
        #    점수가 제일 높은 건 우연히 v3일 수 있는데, 사람이 읽어야 할 건 최신본이다.
        # 계열 키와 본문 해시 둘 다 그룹을 가리킬 수 있다. 한 문서가 해시로
        # 다른 계열 그룹에 붙으면, 그 문서의 계열도 같은 그룹을 가리키게 해 둔다.
        # 안 그러면 뒤에 오는 같은 계열 문서가 새 그룹을 만들어 계열 중복이 생긴다.
        groups: dict[Any, dict[str, Any]] = {}
        order: list[Any] = []
        owner: dict[Any, Any] = {}          # 계열 키 / 본문 해시 -> 그룹 키
        for final, doc_id, idx in scored:
            doc = self.documents[doc_id]
            digest = self._chunk_hash[idx]
            fam = doc.family or doc.title
            key = owner.get(fam) or owner.get(digest) or fam
            group = groups.get(key)
            if group is None:
                group = groups[key] = {"score": final, "members": []}
                order.append(key)
            group["score"] = max(group["score"], final)
            group["members"].append((doc_id, idx, final))
            owner.setdefault(fam, key)
            owner.setdefault(digest, key)

        folded: list[tuple[float, dict[str, Any]]] = []
        for key in order:
            group = groups[key]
            members = group["members"]
            # 대표: 버전이 가장 높은 문서. 같으면 점수, 그래도 같으면 doc_id로 고정한다
            # (실행마다 순서가 흔들리면 gold 라벨이 붙었다 떨어졌다 한다).
            rep_id, rep_idx, _ = max(members, key=lambda m: self._representative_key(m))
            card = self._card(rep_id, rep_idx, group["score"])
            ref = card["source_ref"]
            for doc_id, _idx, _s in members:
                if doc_id == rep_id:
                    continue
                ref["folded_document_ids"].append(doc_id)
                ref["folded_titles"].append(self.documents[doc_id].title)
            ref["folded_count"] = len(ref["folded_document_ids"])
            folded.append((group["score"], card))

        folded.sort(key=lambda item: -item[0])
        return folded[:top_k]

    def _representative_key(self, member: tuple[str, int, float]) -> tuple:
        """계열 안에서 어느 문서를 화면에 세울지 정하는 값. 클수록 대표에 가깝다.

        신호를 세 단계로 쌓는다.

        1. **최종 제출 폴더** — 가장 강하다. 실제로 제출된 산출물이 놓인 자리이고,
           파일명에 버전이 없어도 잡힌다.
        2. **버전 번호** — 파일명의 `_v17`, `_Final`.
        3. **채움 정도** — 표의 빈 셀이 적은 쪽. 같은 버전의 사본 중 더 작성된 것.

        분량(글자 수)은 쓰지 않는다. 실측에서 신호가 아니었다 —
        계획서와 발표자료는 최종본으로 갈수록 다듬어져 **짧아진다**
        (`_v16_jic` 158,468자 > `_v17` 152,923자).
        """
        doc_id, _idx, score = member
        doc = self.documents[doc_id]
        fill = 1.0 - (doc.empty_cells / doc.cells) if doc.cells else 0.0
        return (doc.final_dir, version_rank(doc.title), round(fill, 3), score, doc_id)

    def _filter_weight(
        self, doc: CorpusDocument, filters: dict[str, Any], penalty: float
    ) -> float:
        weight = 1.0
        if project := filters.get("project"):
            key = re.sub(r"[\s_-]+", "", str(project).lower())
            mine = re.sub(r"[\s_-]+", "", self.project.lower())
            if key not in {mine, "공통"}:
                weight *= penalty
        if source_type := filters.get("source_type"):
            if doc.source_type != str(source_type).lower():
                weight *= penalty
        if document_types := filters.get("document_types"):
            allowed = {str(item).lower() for item in document_types}
            if doc.source_type not in allowed:
                weight *= penalty
        # status 필터는 걸지 않는다. 과제 문서에는 draft/approved 같은 상태가 없다.
        return weight

    def _card(self, doc_id: str, chunk_idx: int, score: float) -> dict[str, Any]:
        doc = self.documents[doc_id]
        text = self._chunk_text[chunk_idx]
        quote = re.sub(r"\s+", " ", text).strip()
        return {
            "evidence_id": f"ev_ltm_{doc_id}",
            "tier": "LTM",
            "source_type": doc.source_type,
            "title": doc.title,
            "date": doc.modified_at,
            "project": self.project,
            "summary": quote[:220],
            "quote": quote[:220],
            "content_excerpt": quote[:600],
            "source_ref": {
                "document_id": doc_id,
                "path": doc.rel_path,
                # 어느 청크가 걸렸는지 남긴다. 원문 대조에 필요하다.
                "chunk_index": chunk_idx,
                "chunk_count": doc.n_chunks,
                # 이 카드로 접힌 다른 버전들. 평가에서 gold 라벨이 어느 버전을
                # 가리키든 맞출 수 있도록 id를 그대로 남긴다.
                "folded_document_ids": [],
                "folded_titles": [],
                "folded_count": 0,
            },
            "retrieval_score": round(score, 3),
            "permission_scope": "company_internal",
        }

    # ------------------------------------------------------------------ misc

    def source_types(self) -> list[str]:
        return sorted({d.source_type for d in self.documents.values()})

    def stats(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "chunks": len(self._chunk_doc),
            "vocabulary": len(self._postings),
            "source_types": self.source_types(),
        }


def default_corpus_path(project_root: Path) -> Path:
    """org_agent_mvp 옆 datasets/ 폴더에 20200504 파싱 결과를 둔 기본 배치를 가정한 경로."""
    return (
        project_root.parent
        / "datasets"
        / "20200504-doc_rag"
        / "export"
        / "knowledge_service_core_tech_20200504"
        / "chunks.jsonl"
    )
