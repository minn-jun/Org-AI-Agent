from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .normalization import normalize_project_name


PROJECT_RE = re.compile(r"([A-Za-z0-9가-힣]+\s*[-_]?\s*과제)")


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    can_answer_directly: bool
    memory_needed: bool
    memory_weights: dict[str, float]
    query_rewrites: list[str]
    filters: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleBasedQueryAnalyzer:
    """Transparent baseline that can later be replaced by a small LLM or encoder."""

    RECENT_MARKERS = ("아까", "방금", "오늘", "어제", "최근 대화", "이번 회의", "이전 턴")
    MTM_MARKERS = ("회의록", "초안", "보고서", "진행 중", "이번 달", "수정안", "일정표")
    LTM_MARKERS = ("공식", "최종", "기준", "규칙", "지침", "정책", "회사", "승인")
    COMPARISON_MARKERS = ("충돌", "비교", "차이", "달라", "맞아", "일치")
    MEMORY_MARKERS = (
        "일정",
        "담당자",
        "회의",
        "문서",
        "제안서",
        "프로젝트",
        "과제",
        "예산",
        "결정",
        "근거",
    )
    DIRECT_MARKERS = ("개념", "뜻", "일반적으로", "아이디어", "브레인스토밍")

    def analyze(self, query: str, recent_turns: list[dict[str, Any]] | None = None) -> QueryPlan:
        recent_turns = recent_turns or []
        text = query.strip()
        project_match = PROJECT_RE.search(text)
        filters: dict[str, Any] = {}
        if project_match:
            filters["project"] = normalize_project_name(project_match.group(1))

        is_comparison = self._contains(text, self.COMPARISON_MARKERS)
        is_recent = self._contains(text, self.RECENT_MARKERS)
        is_ltm = self._contains(text, self.LTM_MARKERS)
        is_mtm = self._contains(text, self.MTM_MARKERS)
        has_memory_signal = self._contains(text, self.MEMORY_MARKERS)
        refers_to_session = bool(recent_turns) and self._contains(
            text,
            (
                "그거",
                "그 내용",
                "그 일정",
                "그 담당자",
                "그 문서",
                "앞에서",
                "이어서",
                "그러면",
                "아까",
            ),
        )
        previous_query = str(recent_turns[-1].get("user_query", "")) if recent_turns else ""
        if refers_to_session and "project" not in filters:
            previous_project = PROJECT_RE.search(previous_query)
            if previous_project:
                filters["project"] = normalize_project_name(previous_project.group(1))

        memory_needed = has_memory_signal or is_recent or is_mtm or is_ltm or refers_to_session
        can_answer_directly = not memory_needed and self._contains(text, self.DIRECT_MARKERS)
        if not memory_needed and not can_answer_directly:
            can_answer_directly = True

        if is_comparison:
            intent = "memory_comparison"
            weights = {"stm": 0.35, "mtm": 0.25, "ltm": 0.40}
            reason = "최근 정보와 공식 기준을 함께 비교해야 하는 질문"
        elif is_recent or refers_to_session:
            intent = "recent_context_lookup"
            weights = {"stm": 0.65, "mtm": 0.25, "ltm": 0.10}
            reason = "최근 대화나 최신 결정의 확인이 필요한 질문"
        elif is_ltm:
            intent = "official_knowledge_lookup"
            weights = {"stm": 0.10, "mtm": 0.20, "ltm": 0.70}
            reason = "공식 문서 또는 조직 기준 확인이 필요한 질문"
        elif is_mtm:
            intent = "working_document_lookup"
            weights = {"stm": 0.20, "mtm": 0.65, "ltm": 0.15}
            reason = "진행 중 문서와 최근 산출물 확인이 필요한 질문"
        elif memory_needed:
            intent = "organization_memory_lookup"
            weights = {"stm": 0.34, "mtm": 0.43, "ltm": 0.23}
            reason = "조직 메모리 근거가 필요한 사실 질문"
        else:
            intent = "direct_answer"
            weights = {"stm": 0.0, "mtm": 0.0, "ltm": 0.0}
            reason = "조직 메모리 검색 없이 답변 가능한 일반 질문"

        rewrites = [text]
        if refers_to_session and previous_query:
            rewrites.append(f"이전 질문: {previous_query} 후속 질문: {text}")
        if is_recent:
            rewrites.append(f"{text} 최신 결정 일정 담당자")
        if is_comparison:
            rewrites.append(f"{text} 최근 변경 공식 승인 기준")
        if is_ltm:
            rewrites.append(f"{text} 공식 최종 승인 문서")

        return QueryPlan(
            intent=intent,
            can_answer_directly=can_answer_directly,
            memory_needed=memory_needed,
            memory_weights=weights,
            query_rewrites=list(dict.fromkeys(rewrites)),
            filters=filters,
            reason=reason,
        )

    def _contains(self, text: str, markers: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in markers)
