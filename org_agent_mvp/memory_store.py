from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .normalization import normalize_project_name, normalized_project_key


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_]+")

QUERY_EXPANSIONS = {
    "earlier": "아까 오늘 최근",
    "today": "오늘 금일",
    "meeting": "회의 회의록 대화",
    "next": "다음 차기 후속",
    "steps": "일정 action item 액션아이템 담당자 할일",
    "schedule": "일정 기한 마일스톤",
    "deadline": "기한 일정 완료",
    "action": "action item 액션아이템 담당자 할일",
    "items": "항목 할일 담당자",
    "decision": "결정사항 결정 합의",
    "decisions": "결정사항 결정 합의",
    "official": "공식 최종 승인 기준",
    "plan": "계획서 계획 마일스톤",
    "proposal": "제안서 초안",
    "conflict": "충돌 차이 비교",
    "compare": "비교 차이 충돌",
}


@dataclass(frozen=True)
class MemoryDocument:
    tier: str
    path: Path
    metadata: dict[str, Any]
    text: str


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def _expand_query(query: str) -> str:
    tokens = _tokenize(query)
    expansions = [QUERY_EXPANSIONS[token] for token in tokens if token in QUERY_EXPANSIONS]
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


class MemoryStore:
    def __init__(self, root: Path):
        self.root = root
        self.documents = self._load_all()
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

    def retrieve(
        self,
        tier: str,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        filters = filters or {}
        target_tiers = {"stm", "mtm", "ltm"} if tier == "all" else {tier}
        candidates = [
            doc
            for doc in self.documents
            if doc.tier in target_tiers and self._matches_filters(doc, filters)
        ]

        scored = [
            (score, doc)
            for doc in candidates
            if (score := self._score(query, doc)) > 0
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        results = [
            self._evidence_card(doc, score)
            for score, doc in scored[: max(1, min(top_k, 10))]
        ]
        return {
            "query": query,
            "tier": tier,
            "result_count": len(results),
            "results": results,
        }

    def _matches_filters(self, doc: MemoryDocument, filters: dict[str, Any]) -> bool:
        if project := filters.get("project"):
            document_project = normalized_project_key(str(doc.metadata.get("project", "")))
            filter_project = normalized_project_key(str(project))
            if document_project != filter_project:
                return False
        if source_type := filters.get("source_type"):
            if str(doc.metadata.get("source_type", "")).lower() != str(source_type).lower():
                return False
        if status := filters.get("status"):
            if str(doc.metadata.get("status", "")).lower() != str(status).lower():
                return False
        document_types = filters.get("document_types")
        if document_types:
            source = str(doc.metadata.get("source_type", "")).lower()
            allowed = {str(item).lower() for item in document_types}
            if source not in allowed:
                return False
        return True

    def _score(self, query: str, doc: MemoryDocument) -> float:
        expanded_query = _expand_query(query)
        query_tokens = _tokenize(expanded_query)
        if not query_tokens:
            return 0.0
        title = str(doc.metadata.get("title", ""))
        project = str(doc.metadata.get("project", ""))
        source_type = str(doc.metadata.get("source_type", ""))
        normalized_project = normalize_project_name(project)
        project_key = normalized_project_key(project)
        haystack = (
            f"{title}\n{project}\n{normalized_project}\n"
            f"{project_key}\n{source_type}\n{doc.text}"
        )
        hay_tokens = _tokenize(haystack)
        if not hay_tokens:
            return 0.0
        hay_counts: dict[str, int] = {}
        for token in hay_tokens:
            hay_counts[token] = hay_counts.get(token, 0) + 1

        score = 0.0
        for token in query_tokens:
            if token in hay_counts:
                score += 1.0 + math.log1p(hay_counts[token])
            if token in title.lower():
                score += 2.0
            if token in project.lower():
                score += 1.5
        if doc.tier == "stm" and any(
            marker in expanded_query.lower()
            for marker in ["아까", "오늘", "today", "earlier", "방금"]
        ):
            score += 2.0
        if any(
            marker in expanded_query.lower()
            for marker in ["아까", "오늘", "최근", "today", "earlier", "방금"]
        ):
            document_date = self._document_date(doc)
            if document_date:
                age_days = max(0, (self.latest_document_date - document_date).days)
                score += max(0.0, 4.0 - (age_days * 0.5))
        if doc.tier == "ltm" and any(
            marker in expanded_query.lower()
            for marker in ["공식", "최종", "official", "approved", "기준"]
        ):
            score += 2.0
        return score

    def _document_date(self, doc: MemoryDocument) -> date | None:
        raw = str(doc.metadata.get("date", ""))[:10]
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None

    def _evidence_card(self, doc: MemoryDocument, score: float) -> dict[str, Any]:
        quote = str(doc.metadata.get("summary") or self._best_quote(doc.text))
        confidence = min(0.95, 0.35 + (score / 20.0))
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
            "confidence": round(confidence, 2),
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
