"""실제 문서를 LTM으로 붙이는 어댑터.

doc_rag가 파싱한 `chunks.jsonl`을 읽어 MemoryStore와 같은 모양의 근거 카드를 돌려준다.

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

검색은 청크 단위, 근거는 문서 단위다(`search`).
한 문서의 여러 청크가 걸리면 최고점 청크만 대표로 올리고,
그 청크의 본문을 인용문으로 쓴다. 그래야 STM/MTM과 같은
"문서 하나 = 근거 하나" 계약이 유지된다.
페이지 단위 평가처럼 청크 순위 자체가 필요하면 `search_chunks`를 쓴다.

## 코퍼스 메타데이터

검색기는 특정 코퍼스의 폴더 이름이나 파일명 습관을 모른다.
문서 성격·버전 계열·최종본 여부·소속 과제는 **코퍼스 전처리가 채워 넣은 필드**로만 읽는다.
필드가 없으면 그 기능이 꺼질 뿐 검색은 그대로 돈다.

| 필드 | 뜻 | 없을 때 |
|---|---|---|
| `doc_type` | 문서 성격. 필터와 analyzer enum에 쓴다 | `document` |
| `version_group` | 같은 문서의 버전들이 공유하는 키. 같으면 검색 결과에서 한 건으로 접는다 | 제목 (같은 제목끼리만 접힘) |
| `version_rank` | 계열 안 최신 순서 `[최종 여부, 주 번호, 부 번호]` | `[0, 0, 0]` |
| `is_final` | 실제 제출·확정본인가. 묶음 대표를 고를 때 가장 먼저 본다 | `false` |
| `project` | 문서가 속한 과제·프로젝트. 필터와 과제명 가산점에 쓴다 | 빈 문자열 |

값은 청크 `metadata`에 넣거나, 문서 단위로 `chunks.jsonl` 옆 `document_meta.jsonl`
(한 줄에 `{"doc_id": ..., 필드...}`)에 둔다. 둘 다 있으면 `document_meta.jsonl`이 이긴다.
옆 파일로 두면 청크 파일을 다시 쓰지 않으므로 임베딩 캐시(파일 크기·수정시각 키)가 유지된다.
`LTM_DOC_META=none`이면 옆 파일을 읽지 않고, 경로를 주면 그 파일을 읽는다.

청크 `metadata`의 `page_nos`(없으면 `page_no`)는 청크가 걸친 페이지다. 카드의 `source_ref.page_nos`로 나간다.

2026-09-15까지는 20200504 과제 폴더 전용 규칙(폴더 이름 → 문서 유형, 버전 꼬리표 정규식,
최종제출 폴더, 과제명·기관명 프로파일)이 이 파일에 있었다. 지금은 코퍼스 전처리
(`datasets/20200504-doc_rag/scripts/enrich_doc_meta.py`, 저장소 밖)로 옮겼다.

## 제목 가산점 (`RETRIEVER_TITLE_BONUS`)

| 값 | 방식 |
|---|---|
| `add` (기본) | 질의 토큰이 제목에 부분 문자열로 들어 있으면 토큰마다 +2.0. 첫 MVP부터의 값이고 근거는 없다 |
| `none` | 가산점 없음. 제목은 이미 청크 색인(`제목 + 본문`)에 들어가 BM25로 반영된다 |
| `mult` | `점수 × (1 + 0.1 × 제목 일치율)`. 일치율 = IDF가 질의 중앙값 이상인 토큰 중 제목 토큰에 정확히 있는 비율. 최대 10%라 동점에 가까운 후보의 순서만 바꾼다 |

## 최신성

LTM에는 최신성 보정을 걸지 않는다.
파일 수정일은 문서의 시점이 아니라 파일을 마지막에 만진 날이라
"오늘", "최근" 같은 질의에 공식 계획서가 딸려 올라오면 오히려 틀린다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import dense as dense_mod
from .scoring import BM25_SCORERS, Bm25Params, bm25_params, freq_weight, scorer_name
from .tokenizer import tokenize, tokenize_many

#: 문서 성격을 모를 때의 값. 파일 형식(pdf/hwp)은 쓰지 않는다 —
#: analyzer enum이 형식 이름으로 오염되기 때문이다.
DEFAULT_DOC_TYPE = "document"

#: 문서 단위 메타데이터 필드. 청크 metadata나 document_meta.jsonl에서 읽는다.
DOC_FIELDS = ("doc_type", "version_group", "version_rank", "is_final", "project")

#: 청크 파일 옆에 두는 문서 메타데이터 파일 이름.
DOC_META_FILE = "document_meta.jsonl"

#: 제목 가산점. `add` 방식에서 토큰마다 더하는 값 / `mult` 방식의 최대 비율.
TITLE_BONUS = 2.0
TITLE_BONUS_MULT_ALPHA = 0.1
TITLE_BONUS_MODES = ("add", "none", "mult")
DEFAULT_TITLE_BONUS = "none"

#: 과제명 가산점. 문서의 project 필드에 질의 토큰이 들어 있으면 토큰마다 더한다.
PROJECT_BONUS = 1.5

#: 이 수를 넘는 계열에 같은 본문이 나오면 서식으로 보고 감점한다.
#: 20200504 과제 폴더 표본에서 3계열까지는 작업 파일 사이에 복사된 진짜 내용이었고,
#: 4계열부터 계약 문구·서식 지침 같은 것이 나왔다. 방식은 범용이고 기준값은 그 표본에서 왔다.
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


def title_bonus_mode() -> str:
    """기본은 가산점 없음. 2026-09-16 변경 — Allganize dev에서 `add`와 동률이었고,
    에이전트 전체로 보면 LTM 점수만 부풀려 계층 병합을 무너뜨리는 주범이었다(07 문서 2-3절)."""
    name = os.environ.get("RETRIEVER_TITLE_BONUS", DEFAULT_TITLE_BONUS).strip().lower()
    return name if name in TITLE_BONUS_MODES else DEFAULT_TITLE_BONUS


def _chunk_pages(meta: dict[str, Any]) -> tuple[int, ...]:
    pages = meta.get("page_nos")
    if isinstance(pages, list) and pages:
        return tuple(int(p) for p in pages)
    page = meta.get("page_no")
    return (int(page),) if page else ()


@dataclass
class CorpusDocument:
    doc_id: str
    title: str
    rel_path: str
    source_type: str
    stage: str
    modified_at: str
    n_chunks: int
    family: str = ""                          # version_group. 비면 제목으로 묶는다
    final_dir: int = 0                         # is_final이면 1
    version: tuple[int, ...] = (0, 0, 0)       # version_rank
    project: str = ""
    cells: int = 0         # 표 셀 수
    empty_cells: int = 0   # 그중 빈 셀


def load_document_meta(path: Path | None) -> dict[str, dict[str, Any]]:
    """`document_meta.jsonl`을 doc_id별 필드 사전으로 읽는다. 파일이 없으면 빈 사전이다."""
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["doc_id"])] = {k: row[k] for k in DOC_FIELDS if k in row}
    return rows


def _resolve_doc_meta_path(corpus_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return Path(explicit)
    raw = os.environ.get("LTM_DOC_META", "").strip()
    if raw.lower() in {"none", "off", "0"}:
        return None
    return Path(raw) if raw else corpus_path.with_name(DOC_META_FILE)


class LtmCorpus:
    """청크 역색인 + 문서 단위 근거 집계."""

    def __init__(self, path: Path, doc_meta_path: Path | None = None):
        self.path = Path(path)
        self.doc_meta_path = _resolve_doc_meta_path(self.path, doc_meta_path)
        self.documents: dict[str, CorpusDocument] = {}

        # 청크는 인덱스 정렬 배열로 들고 있는다. 청크마다 dict를 만들면
        # 26,586개 × 토큰 dict라 메모리가 수백 MB로 뛴다.
        self._chunk_doc: list[str] = []      # 청크 -> doc_id
        self._chunk_text: list[str] = []
        self._chunk_title_l: list[str] = []
        self._chunk_hash: list[bytes] = []   # 본문 해시. 버전 간 동일 문단을 접는 데 쓴다
        self._chunk_len: list[int] = []      # 청크 토큰 수. BM25 길이 정규화에 쓴다
        self._chunk_pages: list[tuple[int, ...]] = []   # 청크가 걸친 페이지

        # token -> array('i') [청크번호, 빈도, 청크번호, 빈도, ...]
        self._postings: dict[str, array] = {}

        # 청크 본문이 몇 개의 *계열*에 걸쳐 나오는지. 보일러플레이트 감쇠에 쓴다.
        self._chunk_weight: list[float] = []

        # 제목 토큰 집합. `mult` 제목 가산점을 처음 쓸 때 만든다.
        self._title_tokens: dict[str, frozenset[str]] | None = None

        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        postings: dict[str, array] = {}
        doc_meta = load_document_meta(self.doc_meta_path)
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                meta = row["metadata"]
                doc_id = row["source_id"]
                if doc_id not in self.documents:
                    fields = {k: meta[k] for k in DOC_FIELDS if k in meta}
                    fields.update(doc_meta.get(doc_id, {}))
                    self.documents[doc_id] = CorpusDocument(
                        doc_id=doc_id,
                        title=meta.get("title") or doc_id,
                        rel_path=meta.get("source_path", ""),
                        source_type=str(fields.get("doc_type") or DEFAULT_DOC_TYPE),
                        stage=meta.get("stage", ""),
                        modified_at=str(meta.get("modified_at", ""))[:10],
                        n_chunks=0,
                        family=str(fields.get("version_group") or ""),
                        final_dir=1 if fields.get("is_final") else 0,
                        version=tuple(int(v) for v in (fields.get("version_rank") or (0, 0, 0))),
                        project=str(fields.get("project") or ""),
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
                self._chunk_pages.append(_chunk_pages(meta))

        # 토큰화는 한 건씩 부르지 않고 묶어서 넘긴다.
        # 형태소 분석기는 호출 비용이 커서 차이가 크다 — 실측으로 26,031청크에
        # 개별 호출 158초, 배치 호출 14초였다(11.5배).
        for start in range(0, len(self._chunk_text), _TOKENIZE_BATCH):
            stop = start + _TOKENIZE_BATCH
            # 제목과 본문을 **따로** 자른 뒤 합친다. 이어 붙여서 한 번에 자르면
            # 형태소 분석기가 경계에서 문맥을 잘못 읽는다 — 2026-09-16 실측:
            #   "이월 신청 내용"        -> ['이월', '신청', '내용']
            #   "발췌 메모 이월 신청 내용" -> ['발췌', '메모', '월', '신청', '내용']   ('이'가 지시어로 붙는다)
            # 본문 첫 단어가 조용히 다른 토큰이 되어 질의와 만나지 못한다.
            span = range(start, min(stop, len(self._chunk_text)))
            titles = [self.documents[self._chunk_doc[i]].title for i in span]
            bodies = [self._chunk_text[i] for i in span]
            merged = [t + b for t, b in zip(tokenize_many(titles), tokenize_many(bodies))]
            for offset, tokens in enumerate(merged):
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
        # k1과 BM25+ 여부는 검색할 때 설정을 읽어 정한다(scoring.bm25_params).
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
        20200504 과제 폴더 실측으로 청크의 9.4%가 4개 이상 계열에 걸쳐 있었고,
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

    def _score_chunks(self, query_tokens: list[str], query_text: str) -> dict[int, float]:
        """청크별 점수 (본문 일치 × 서식 감쇠 → 임베딩 융합 → 제목·과제명 가산점)."""
        # 1) 질의어가 든 청크만 훑는다. 없는 토큰은 posting이 아예 없다.
        #    점수 함수는 scoring 모듈이 정한다(RETRIEVER_SCORER).
        #    BM25는 토큰마다 df가 필요한데, posting 길이가 곧 df라 공짜로 얻는다.
        chunk_score: dict[int, float] = defaultdict(float)
        use_bm25 = scorer_name() in BM25_SCORERS
        params = bm25_params(self._bm25.n_docs, self._bm25.avg_len) if use_bm25 else None
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
            return chunk_score

        # 2) 제목 가중치는 후보가 정해진 뒤에만 본다. 방식은 RETRIEVER_TITLE_BONUS.
        #    과제명은 문서마다 다를 수 있어 문서의 project 필드로 계산한다.
        mode = title_bonus_mode()
        lowered = list(dict.fromkeys(query_tokens))
        title_ratio = self._title_match_ratio(lowered) if mode == "mult" else None
        project_bonus_of: dict[str, float] = {}
        for idx, base in list(chunk_score.items()):
            doc_id = self._chunk_doc[idx]
            if mode == "add":
                title_l = self._chunk_title_l[idx]
                value = base + TITLE_BONUS * sum(1 for t in lowered if t in title_l)
            elif mode == "mult":
                value = base * (1.0 + TITLE_BONUS_MULT_ALPHA * title_ratio(doc_id))
            else:
                value = base
            project = self.documents[doc_id].project
            project_bonus = project_bonus_of.get(project)
            if project_bonus is None:
                project_l = project.lower()
                project_bonus = project_bonus_of[project] = PROJECT_BONUS * sum(1 for t in lowered if t in project_l)
            chunk_score[idx] = value + project_bonus

        # 3) 재정렬기(RETRIEVER_RERANK)를 켰으면 상위 N개의 순서만 다시 매긴다.
        #    점수 값은 그대로 두고 재정렬 순서대로 나눠 주므로 prefetch 병합의 눈금은 변하지 않는다.
        from . import rerank as rerank_mod

        scorer = rerank_mod.get_scorer()
        if scorer is not None:
            query = query_text or " ".join(lowered)
            chunk_score = rerank_mod.reorder(
                chunk_score, query,
                lambda idx: f"{self.documents[self._chunk_doc[idx]].title}\n{self._chunk_text[idx]}",
                scorer, rerank_mod.top_n(),
            )
        return chunk_score

    def _title_match_ratio(self, query_tokens: list[str]) -> Callable[[str], float]:
        """`mult` 제목 가산점의 일치율. 흔한 토큰(IDF가 질의 중앙값 미만)은 세지 않는다."""
        present = [t for t in query_tokens if t in self._postings]
        if not present:
            return lambda doc_id: 0.0
        idf = {t: self._bm25.idf(len(self._postings[t]) // 2) for t in present}
        cut = statistics.median(idf.values())
        keep = [t for t in present if idf[t] >= cut]
        if self._title_tokens is None:
            docs = list(self.documents.values())
            self._title_tokens = {
                doc.doc_id: frozenset(tokens)
                for doc, tokens in zip(docs, tokenize_many([doc.title for doc in docs]))
            }
        titles = self._title_tokens
        cache: dict[str, float] = {}

        def ratio(doc_id: str) -> float:
            if doc_id not in cache:
                cache[doc_id] = sum(1 for t in keep if t in titles[doc_id]) / len(keep)
            return cache[doc_id]

        return ratio

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
        chunk_score = self._score_chunks(query_tokens, query_text)
        if not chunk_score:
            return []

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

    def search_chunks(
        self,
        query_tokens: list[str],
        *,
        query_text: str = "",
        top_k: int = 10,
    ) -> list[tuple[float, dict[str, Any]]]:
        """청크 단위 순위. 문서로 접지 않고 필터도 걸지 않는다. 페이지 단위 평가용이다.

        점수는 `search`와 같다(같은 `_score_chunks`). 동점은 청크가 점수에 들어간 순서를 따른다.
        """
        if not query_tokens:
            return []
        chunk_score = self._score_chunks(query_tokens, query_text)
        ranked = sorted(chunk_score.items(), key=lambda item: -item[1])[:top_k]
        return [(score, self._chunk_card(idx, score)) for idx, score in ranked]

    def _representative_key(self, member: tuple[str, int, float]) -> tuple:
        """계열 안에서 어느 문서를 화면에 세울지 정하는 값. 클수록 대표에 가깝다.

        신호를 세 단계로 쌓는다. 앞의 두 값은 코퍼스 전처리가 채운 메타데이터다.

        1. **확정본 여부** (`is_final`) — 가장 강하다. 실제로 제출된 산출물을 가리킨다.
        2. **버전 순서** (`version_rank`) — 파일명의 `_v17`, `_Final` 같은 표기에서 온다.
        3. **채움 정도** — 표의 빈 셀이 적은 쪽. 같은 버전의 사본 중 더 작성된 것.

        분량(글자 수)은 쓰지 않는다. 실측에서 신호가 아니었다 —
        계획서와 발표자료는 최종본으로 갈수록 다듬어져 **짧아진다**
        (`_v16_jic` 158,468자 > `_v17` 152,923자).
        """
        doc_id, _idx, score = member
        doc = self.documents[doc_id]
        fill = 1.0 - (doc.empty_cells / doc.cells) if doc.cells else 0.0
        return (doc.final_dir, doc.version, round(fill, 3), score, doc_id)

    def _filter_weight(
        self, doc: CorpusDocument, filters: dict[str, Any], penalty: float
    ) -> float:
        weight = 1.0
        if project := filters.get("project"):
            key = re.sub(r"[\s_-]+", "", str(project).lower())
            mine = re.sub(r"[\s_-]+", "", doc.project.lower())
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
            "project": doc.project,
            "summary": quote[:220],
            "quote": quote[:220],
            "content_excerpt": quote[:600],
            "source_ref": {
                "document_id": doc_id,
                "path": doc.rel_path,
                # 어느 청크가 걸렸는지 남긴다. 원문 대조에 필요하다.
                "chunk_index": chunk_idx,
                "chunk_count": doc.n_chunks,
                "page_nos": list(self._chunk_pages[chunk_idx]),
                # 이 카드로 접힌 다른 버전들. 평가에서 gold 라벨이 어느 버전을
                # 가리키든 맞출 수 있도록 id를 그대로 남긴다.
                "folded_document_ids": [],
                "folded_titles": [],
                "folded_count": 0,
            },
            "retrieval_score": round(score, 3),
            "permission_scope": "company_internal",
        }

    def _chunk_card(self, chunk_idx: int, score: float) -> dict[str, Any]:
        doc = self.documents[self._chunk_doc[chunk_idx]]
        return {
            "evidence_id": f"ev_ltm_{doc.doc_id}",
            "document_id": doc.doc_id,
            "title": doc.title,
            "path": doc.rel_path,
            "chunk_index": chunk_idx,
            "page_nos": list(self._chunk_pages[chunk_idx]),
            "retrieval_score": round(score, 3),
            "quote": re.sub(r"\s+", " ", self._chunk_text[chunk_idx]).strip()[:220],
        }

    # ------------------------------------------------------------------ misc

    def source_types(self) -> list[str]:
        return sorted({d.source_type for d in self.documents.values()})

    def projects(self) -> list[str]:
        return sorted({d.project for d in self.documents.values() if d.project})

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
