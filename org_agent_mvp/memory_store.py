from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .ltm_corpus import LtmCorpus
from .normalization import normalize_project_name, normalized_project_key
from .scoring import Bm25Params, freq_weight, scorer_name
from .tokenizer import tokenize, tokenize_many

# 키는 부분 문자열로 매칭한다. 한국어는 조사가 붙어 토큰이 달라지므로
# ("예산이" != "예산") 정확 일치로 조회하면 확장이 거의 동작하지 않는다.
QUERY_EXPANSIONS = {
    # 시점
    "아까": "오늘 최근 방금 회의 earlier",
    "방금": "아까 오늘 최근 earlier",
    "오늘": "금일 최근 today",
    "어제": "전날 최근",
    "최근": "오늘 최신 근래 recent",
    # 회의와 대화
    "회의": "회의록 미팅 대화 논의 meeting",
    "대화": "회의 논의 conversation",
    # 일정
    "일정": "기한 마감 마일스톤 날짜 schedule",
    "기한": "일정 마감 완료일 deadline",
    "마감": "기한 일정 deadline",
    "다음": "차기 후속 next",
    # 담당과 실행
    "담당": "담당자 책임 배정 owner",
    "action": "action item 액션아이템 할일 담당자",
    "액션": "action item 할일 담당자",
    "할일": "action item 액션아이템 담당자",
    # 예산
    "예산": "사업비 비용 단가 산정 자문비 budget",
    "비용": "예산 단가 산정 cost",
    "단가": "예산 비용 산정 기준",
    "산정": "예산 기준 근거 산출",
    # 결정과 공식성
    "결정": "결정사항 합의 확정 decision",
    "공식": "최종 승인 확정 기준 official approved",
    "최종": "공식 승인 확정 final",
    "승인": "공식 최종 확정 approved",
    "기준": "규정 지침 표준 standard",
    "계획": "계획서 마일스톤 일정 plan",
    # 문서
    "제안서": "초안 제안 proposal",
    "초안": "제안서 작성중 draft",
    "보고서": "보고 리포트 report",
    # 비교와 충돌
    "충돌": "차이 비교 불일치 conflict",
    "비교": "차이 충돌 대조 compare",
    "차이": "비교 충돌 불일치",
    # 리스크
    "리스크": "위험 지연 쟁점 risk",
    "지연": "리스크 위험 delay",
    "쟁점": "이슈 논점 리스크 issue",
}

#: retrieve()가 한 번에 돌려줄 수 있는 최대 건수. A방식의 넓은 후보 풀을 위해 상향했다.
MAX_RETRIEVE_TOP_K = 50


@dataclass(frozen=True)
class MemoryDocument:
    tier: str
    path: Path
    metadata: dict[str, Any]
    text: str


def _tokenize(text: str) -> list[str]:
    """검색 토큰화. 방식은 tokenizer 모듈이 정한다(RETRIEVER_TOKENIZER).

    질의와 문서가 같은 함수를 타야 매칭이 된다.
    LTM(ltm_corpus)도 같은 모듈을 쓴다.
    """
    return tokenize(text)


def _expand_query(query: str) -> str:
    lowered = query.lower()
    expansions = [value for key, value in QUERY_EXPANSIONS.items() if key in lowered]
    return " ".join([query, *expansions])


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip('"').strip("'") for part in inner.split(",")]
    return value.strip('"').strip("'")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    meta_text = text[4:end].strip()
    body = text[end + 4 :].lstrip()
    metadata: dict[str, Any] = {}
    for line in meta_text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = _parse_scalar(value)
    return metadata, body


def _load_document(path: Path, tier: str) -> MemoryDocument:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        metadata = {k: v for k, v in data.items() if k not in {"body", "content"}}
        body = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        metadata, body = _parse_frontmatter(text)
    metadata.setdefault("memory_tier", tier)
    metadata.setdefault("title", path.stem)
    metadata.setdefault("source_id", path.name)
    return MemoryDocument(tier=tier, path=path, metadata=metadata, text=body)


#: 필터가 어긋난 문서에 곱하는 감점 계수.
#: 0.0이면 이전의 하드 필터와 같고, 1.0이면 필터를 무시한다.
DEFAULT_FILTER_PENALTY = 0.3


def build_memory_store(config, filter_penalty: float | None = None) -> "MemoryStore":
    """설정대로 MemoryStore를 만든다.

    `config.ltm_corpus_path`가 있으면 실제 과제 문서를 LTM으로 붙인다.
    코퍼스 로드는 5초쯤 걸리므로(26,586청크 역색인) 실행마다 한 번만 한다.
    """
    corpus = LtmCorpus(config.ltm_corpus_path) if config.ltm_corpus_path else None
    return MemoryStore(
        config.memory_root,
        filter_penalty=config.filter_penalty if filter_penalty is None else filter_penalty,
        ltm_corpus=corpus,
    )


class MemoryStore:
    def __init__(
        self,
        root: Path,
        filter_penalty: float = DEFAULT_FILTER_PENALTY,
        ltm_corpus: "LtmCorpus | None" = None,
    ):
        self.root = root
        self.filter_penalty = filter_penalty
        # 실제 과제 문서를 LTM으로 붙일 때만 들어온다. 없으면 기존처럼
        # `ltm/` 폴더의 파일만 LTM으로 쓴다.
        self.ltm_corpus = ltm_corpus
        self.documents = self._load_all()
        self._doc_index = self._index_documents()
        # BM25용 코퍼스 통계. LTM과 같은 식을 쓴다.
        self._doc_freq: dict[str, int] = {}
        for entry in self._doc_index.values():
            for token in entry["counts"]:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
        lengths = [e["n_tokens"] for e in self._doc_index.values()] or [1]
        self._bm25 = Bm25Params(n_docs=len(lengths), avg_len=sum(lengths) / len(lengths))
        document_dates = [
            parsed
            for doc in self.documents
            if (parsed := self._document_date(doc)) is not None
        ]
        self.latest_document_date = max(document_dates, default=date.today())

    def _load_all(self) -> list[MemoryDocument]:
        docs: list[MemoryDocument] = []
        for tier in ("stm", "mtm", "ltm"):
            tier_dir = self.root / tier
            if not tier_dir.exists():
                continue
            for path in sorted(tier_dir.glob("*")):
                if path.suffix.lower() not in {".md", ".json", ".txt"}:
                    continue
                docs.append(_load_document(path, tier))
        return docs

    def _index_documents(self) -> dict[int, dict[str, Any]]:
        """문서별 토큰 빈도를 미리 세 둔다.

        `_score()`가 질의마다 문서를 다시 자르던 것을 로드 시점으로 옮긴 것이다.
        형태소 분석기를 쓰면 자르는 비용이 커서 차이가 크다.

        haystack에 제목·과제명·문서 유형을 같이 넣는 이유는 예전 그대로다 —
        본문에 없어도 제목에 있으면 걸리게 하려는 것이다.
        """
        haystacks: list[str] = []
        for doc in self.documents:
            title = str(doc.metadata.get("title", ""))
            project = str(doc.metadata.get("project", ""))
            source_type = str(doc.metadata.get("source_type", ""))
            parts = [
                title,
                project,
                normalize_project_name(project),
                normalized_project_key(project),
                source_type,
                doc.text,
            ]
            haystacks.append("\n".join(parts))

        index: dict[int, dict[str, Any]] = {}
        for doc, tokens in zip(self.documents, tokenize_many(haystacks)):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            index[id(doc)] = {
                "counts": counts,
                "n_tokens": len(tokens),
                "title_l": str(doc.metadata.get("title", "")).lower(),
                "project_l": str(doc.metadata.get("project", "")).lower(),
            }
        return index

    def retrieve(
        self,
        tier: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        filters = filters or {}
        target_tiers = {"stm", "mtm", "ltm"} if tier == "all" else {tier}
        prepared = self._prepare_query(query)

        # 필터를 하드 컷이 아니라 점수 배수로 적용한다.
        # 하드 필터는 값이 하나만 어긋나도 후보가 0건이 되어, analyzer가 만든
        # 값 하나 때문에 검색 전체가 실패하는 일이 실제로 있었다.
        scored: list[tuple[float, MemoryDocument]] = []
        for doc in self.documents:
            if doc.tier not in target_tiers:
                continue
            weight = self._filter_weight(doc, filters)
            if weight <= 0:
                continue
            score = self._score(query, doc, prepared) * weight
            if score > 0:
                scored.append((score, doc))
        limit = max(1, min(top_k, MAX_RETRIEVE_TOP_K))
        scored.sort(key=lambda item: item[0], reverse=True)
        cards: list[tuple[float, dict[str, Any]]] = [
            (score, self._evidence_card(doc, score)) for score, doc in scored[:limit]
        ]

        # 실코퍼스 LTM은 파일이 아니라 청크 역색인에서 온다. 점수 체계가 같아서
        # 같은 목록에 넣고 다시 정렬하면 된다. (ltm_corpus.py 참고)
        if self.ltm_corpus is not None and "ltm" in target_tiers:
            cards.extend(
                self.ltm_corpus.search(
                    prepared[0],
                    # 임베딩 검색은 토큰이 아니라 원문 문장이 필요하다.
                    query_text=prepared[1],
                    filters=filters,
                    filter_penalty=self.filter_penalty,
                    top_k=limit,
                )
            )
            cards.sort(key=lambda item: item[0], reverse=True)
            cards = cards[:limit]

        results = [card for _, card in cards]
        return {
            "query": query,
            "tier": tier,
            "result_count": len(results),
            "results": results,
        }

    def filter_vocabulary(self) -> dict[str, list[str]]:
        """필터에 실제로 쓸 수 있는 값 목록.

        analyzer 스키마의 enum을 여기서 만든다. 코퍼스에 없는 값을 LLM이 지어내면
        하드 필터에 걸려 후보가 0건이 되므로, 애초에 만들 수 없게 막는 편이 낫다.
        코퍼스가 바뀌면 목록도 따라 바뀐다.
        """
        projects = {
            str(doc.metadata.get("project", "")).strip()
            for doc in self.documents
        }
        source_types = {
            str(doc.metadata.get("source_type", "")).strip()
            for doc in self.documents
        }
        if self.ltm_corpus is not None:
            projects.add(self.ltm_corpus.project)
            source_types.update(self.ltm_corpus.source_types())
        return {
            "project": sorted(value for value in projects if value),
            "source_type": sorted(value for value in source_types if value),
        }

    def _filter_weight(self, doc: MemoryDocument, filters: dict[str, Any]) -> float:
        """필터 일치도를 점수 배수로 돌려준다.

        일치하면 1.0, 어긋나면 `filter_penalty`를 곱한다.
        여러 필터가 동시에 어긋나면 계속 곱해져 더 강하게 밀린다.
        `filter_penalty = 0.0`이면 이전의 하드 필터와 동작이 같다.

        `공통` 문서는 조직 전체에 적용되는 기준이므로 어느 과제 질문에서든
        감점하지 않는다.
        """
        weight = 1.0
        if project := filters.get("project"):
            document_project = normalized_project_key(str(doc.metadata.get("project", "")))
            filter_project = normalized_project_key(str(project))
            if document_project not in {filter_project, normalized_project_key("공통")}:
                weight *= self.filter_penalty
        if source_type := filters.get("source_type"):
            if str(doc.metadata.get("source_type", "")).lower() != str(source_type).lower():
                weight *= self.filter_penalty
        if status := filters.get("status"):
            if str(doc.metadata.get("status", "")).lower() != str(status).lower():
                weight *= self.filter_penalty
        document_types = filters.get("document_types")
        if document_types:
            source = str(doc.metadata.get("source_type", "")).lower()
            allowed = {str(item).lower() for item in document_types}
            if source not in allowed:
                weight *= self.filter_penalty
        return weight

    def _prepare_query(self, query: str) -> tuple[list[str], str]:
        """쿼리 확장과 토큰화를 한 번만 수행한다.

        이전에는 _score() 안에서 문서마다 다시 계산했다. 쿼리 확장은 문서와
        무관하므로 후보 수만큼 낭비된다. 문서 115건에서는 8ms 수준이지만
        실코퍼스(수천 chunk)로 가면 그대로 비례해 늘어난다.
        """
        expanded = _expand_query(query)
        # 확장 결과에는 같은 토큰이 여러 번 들어온다. QUERY_EXPANSIONS의 값이
        # 서로 겹치기 때문이다 — "예산 비용 단가"를 확장하면 세 항목이 모두
        # "예산 단가 산정"을 덧붙여 각 토큰이 3번씩 나온다.
        #
        # 중복을 그대로 두면 그 토큰의 배점이 3배가 된다. 동의어 표가 어쩌다
        # 그렇게 적혀 있다는 것 말고는 근거가 없는 가중치다. 게다가 LTM은
        # set(query_tokens)로 훑어서 중복을 세지 않으므로, 같은 질의에
        # 계층마다 다른 눈금이 적용되고 prefetch의 전역 정규화가 어긋난다.
        #
        # 순서는 유지하면서 중복만 없앤다(dict는 삽입 순서를 지킨다).
        return list(dict.fromkeys(_tokenize(expanded))), expanded.lower()

    def _score(
        self,
        query: str,
        doc: MemoryDocument,
        prepared: tuple[list[str], str] | None = None,
    ) -> float:
        query_tokens, expanded_query = prepared or self._prepare_query(query)
        if not query_tokens:
            return 0.0
        # 문서 쪽 토큰은 로드할 때 미리 세 둔다(_index_documents).
        # 질의마다 다시 자르면 형태소 분석기에서 비용이 폭발한다 —
        # 실측으로 문서 85건에 질의당 247ms였다.
        indexed = self._doc_index.get(id(doc))
        if indexed is None:
            return 0.0
        hay_counts = indexed["counts"]
        if not hay_counts:
            return 0.0
        title_l = indexed["title_l"]
        project_l = indexed["project_l"]

        score = 0.0
        use_bm25 = scorer_name("seed") == "bm25"
        # query_tokens는 _prepare_query에서 이미 중복이 제거돼 있다.
        for token in query_tokens:
            if token in hay_counts:
                if use_bm25:
                    score += self._bm25.weight(
                        hay_counts[token], indexed["n_tokens"], self._doc_freq.get(token, 1)
                    )
                else:
                    score += freq_weight(hay_counts[token])
            if token in title_l:
                score += 2.0
            if token in project_l:
                score += 1.5
        # tier 자체에 주는 보정은 여기에 두지 않는다.
        # tier 선호는 prefetch의 prior에서 한 번만 적용한다. 여기서 또 더하면
        # 같은 신호가 중복 계산되고, alpha=0 실험군에서도 tier 효과가 남는다.
        # 문서 날짜 기반 최신성(Recency)은 tier와 독립적인 요소이므로 유지한다.
        if any(
            marker in expanded_query
            for marker in ["아까", "오늘", "최근", "today", "earlier", "방금"]
        ):
            document_date = self._document_date(doc)
            if document_date:
                age_days = max(0, (self.latest_document_date - document_date).days)
                score += max(0.0, 4.0 - (age_days * 0.5))
        return score

    def _document_date(self, doc: MemoryDocument) -> date | None:
        raw = str(doc.metadata.get("date", ""))[:10]
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None

    def _evidence_card(self, doc: MemoryDocument, score: float) -> dict[str, Any]:
        quote = str(doc.metadata.get("summary") or self._best_quote(doc.text))
        # confidence는 캘리브레이션 근거가 없는 임의 상수에서 나온 값이라 제거했다.
        # 모델이 이를 신뢰도로 읽으면 판단을 오도한다. 순위는 final_score로 표현한다.
        return {
            "evidence_id": f"ev_{doc.tier}_{doc.path.stem}",
            "tier": doc.tier.upper(),
            "source_type": doc.metadata.get("source_type", "unknown"),
            "title": doc.metadata.get("title", doc.path.stem),
            "date": doc.metadata.get("date", ""),
            "project": doc.metadata.get("project", ""),
            "summary": doc.metadata.get("summary", quote),
            "quote": quote,
            "content_excerpt": self._content_excerpt(doc.text),
            "source_ref": {
                "document_id": doc.path.name,
                "path": str(doc.path.relative_to(self.root)),
            },
            "retrieval_score": round(score, 3),
            "permission_scope": doc.metadata.get("permission_scope", "internal"),
        }

    def _best_quote(self, text: str) -> str:
        lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if len(line) >= 15:
                return line[:220]
        return (lines[0] if lines else "")[:220]

    def _content_excerpt(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        return compact[:600]
