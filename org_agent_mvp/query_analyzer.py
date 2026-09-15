from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .normalization import normalize_project_name, normalized_project_key


PROJECT_RE = re.compile(r"([A-Za-z0-9가-힣]+\s*[-_]?\s*(?:과제|프로젝트|사업|과업))")


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    can_answer_directly: bool
    memory_needed: bool
    answer_source: str
    use_session_context: bool
    memory_weights: dict[str, float]
    query_rewrites: list[str]
    filters: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QueryAnalyzer(Protocol):
    def analyze(self, query: str, recent_turns: list[dict[str, Any]] | None = None) -> QueryPlan:
        ...


class RuleBasedQueryAnalyzer:
    """Transparent baseline that can later be replaced by a small LLM or encoder."""

    def __init__(self, vocabulary: dict[str, list[str]] | None = None):
        """`vocabulary`를 주면 코퍼스에 실제로 있는 과제명만 필터로 인정한다.

        PROJECT_RE는 "한 단어 + 과제/사업/프로젝트"를 잡는데, 일반 명사도 걸린다.
        실측으로 잡힌 오탐들이다.

            "총사업비와 순현재가치가..."        -> "총 사업"
            "이 과제의 전체 연구개발기간은..."   -> "이 과제"
            "연구개발과제번호가 뭐야?"          -> "연구개발 과제"
            "산업기술혁신사업 공통 운영요령..."  -> "산업기술혁신 사업"

        이 값이 필터로 들어가면 모든 문서가 감점되어 검색이 통째로 망가진다.
        실제로 평가셋에서 두 케이스가 이것 때문에 정답을 하나도 못 찾았다.

        vocabulary가 없으면 예전대로 동작한다(기존 테스트 호환).
        """
        projects = (vocabulary or {}).get("project") or []
        self._known_projects = {normalized_project_key(p) for p in projects if p}

    RECENT_MARKERS = ("아까", "방금", "오늘", "어제", "최근", "최근 대화", "이번 회의", "이전 턴")
    MTM_MARKERS = ("회의록", "초안", "보고서", "진행 중", "이번 달", "수정안", "일정표")
    LTM_MARKERS = ("공식", "최종", "기준", "규칙", "지침", "정책", "회사", "승인")
    COMPARISON_MARKERS = ("충돌", "비교", "차이", "달라", "다른", "어긋", "맞아", "일치")
    SESSION_REFERENCE_MARKERS = (
        "그거",
        "그 내용",
        "그 일정",
        "그 담당자",
        "그 문서",
        "여기서",
        "위 내용",
        "위에서",
        "앞에서",
        "앞 답변",
        "방금 답변",
        "이어서",
        "그러면",
        "해당",
        "이 내용",
        "이 일정",
        "아까",
    )
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
    SESSION_SUMMARY_MARKERS = ("요약", "정리", "다시 말", "말한거", "말한 것", "정리해줘")

    def analyze(self, query: str, recent_turns: list[dict[str, Any]] | None = None) -> QueryPlan:
        recent_turns = recent_turns or []
        text = query.strip()
        project_match = PROJECT_RE.search(text)
        filters: dict[str, Any] = {}
        if project_match:
            candidate = normalize_project_name(project_match.group(1))
            if self._is_known_project(candidate):
                filters["project"] = candidate

        is_comparison = self._contains(text, self.COMPARISON_MARKERS)
        is_recent = self._contains(text, self.RECENT_MARKERS)
        is_ltm = self._contains(text, self.LTM_MARKERS)
        is_mtm = self._contains(text, self.MTM_MARKERS)
        has_memory_signal = self._contains(text, self.MEMORY_MARKERS)
        refers_to_session = bool(recent_turns) and self._contains(
            text,
            self.SESSION_REFERENCE_MARKERS,
        )
        is_session_summary = (
            refers_to_session
            and self._contains(text, self.SESSION_SUMMARY_MARKERS)
            and not has_memory_signal
            and not is_comparison
            and not is_mtm
            and not is_ltm
        )
        memory_needed = (
            not is_session_summary
            and (has_memory_signal or is_recent or is_mtm or is_ltm or refers_to_session)
        )
        can_answer_directly = not memory_needed and self._contains(text, self.DIRECT_MARKERS)
        if is_session_summary:
            can_answer_directly = True
        elif not memory_needed and not can_answer_directly:
            # 신호가 하나도 없으면 검색 쪽으로 기운다.
            # 불필요한 검색은 토큰을 더 쓸 뿐이지만, 필요한 검색을 건너뛰면
            # 근거 없이 답하게 되어 실패 방향이 훨씬 나쁘다.
            # 일반 개념 질문은 위의 DIRECT_MARKERS에서 이미 걸러진다.
            memory_needed = True

        previous_query = str(recent_turns[-1].get("user_query", "")) if recent_turns else ""
        should_inherit_project = bool(recent_turns) and memory_needed
        if should_inherit_project and "project" not in filters:
            previous_project = self._latest_project_from_turns(recent_turns)
            if previous_project:
                filters["project"] = previous_project

        if is_session_summary:
            intent = "session_context_answer"
            weights = {"stm": 0.0, "mtm": 0.0, "ltm": 0.0}
            reason = "이전 대화 맥락만으로 답변 가능한 세션 후속 질문"
            answer_source = "session_only"
        elif is_comparison:
            intent = "memory_comparison"
            weights = {"stm": 0.35, "mtm": 0.25, "ltm": 0.40}
            reason = "최근 정보와 공식 기준을 함께 비교해야 하는 질문"
            answer_source = "memory_prefetch"
        elif is_recent or refers_to_session:
            intent = "recent_context_lookup"
            weights = {"stm": 0.65, "mtm": 0.25, "ltm": 0.10}
            reason = "최근 대화나 최신 결정의 확인이 필요한 질문"
            answer_source = "memory_prefetch"
        elif is_ltm:
            intent = "official_knowledge_lookup"
            weights = {"stm": 0.10, "mtm": 0.20, "ltm": 0.70}
            reason = "공식 문서 또는 조직 기준 확인이 필요한 질문"
            answer_source = "memory_prefetch"
        elif is_mtm:
            intent = "working_document_lookup"
            weights = {"stm": 0.20, "mtm": 0.65, "ltm": 0.15}
            reason = "진행 중 문서와 최근 산출물 확인이 필요한 질문"
            answer_source = "memory_prefetch"
        elif memory_needed:
            intent = "organization_memory_lookup"
            weights = {"stm": 0.34, "mtm": 0.43, "ltm": 0.23}
            reason = "조직 메모리 근거가 필요한 사실 질문"
            answer_source = "memory_prefetch"
        else:
            intent = "direct_answer"
            weights = {"stm": 0.0, "mtm": 0.0, "ltm": 0.0}
            reason = "조직 메모리 검색 없이 답변 가능한 일반 질문"
            answer_source = "direct_answer"

        # query_rewrites에는 tier 선호 신호를 넣지 않는다.
        # "공식 최종 승인" 같은 단어를 검색 쿼리에 주입하면 제목에 그 단어가 있는 문서가
        # 주제 관련성과 무관하게 상위로 올라온다. tier 선호는 memory_weights(prior)
        # 한 곳에서만 적용한다.
        rewrites = [text]
        if refers_to_session and previous_query:
            project_context = f" 프로젝트: {filters['project']}" if "project" in filters else ""
            rewrites.append(f"이전 질문: {previous_query}{project_context} 후속 질문: {text}")

        return QueryPlan(
            intent=intent,
            can_answer_directly=can_answer_directly,
            memory_needed=memory_needed,
            answer_source=answer_source,
            use_session_context=bool(recent_turns) and (refers_to_session or is_session_summary),
            memory_weights=weights,
            query_rewrites=list(dict.fromkeys(rewrites)),
            filters=filters,
            reason=reason,
        )

    def _is_known_project(self, name: str) -> bool:
        """코퍼스에 있는 과제명만 통과시킨다. 목록이 없으면 전부 통과(예전 동작)."""
        if not self._known_projects:
            return True
        return normalized_project_key(name) in self._known_projects

    def _contains(self, text: str, markers: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(marker.lower() in lowered for marker in markers)

    def _latest_project_from_turns(self, recent_turns: list[dict[str, Any]]) -> str:
        for turn in reversed(recent_turns):
            cached_project = (
                turn.get("query_analysis", {})
                .get("filters", {})
                .get("project", "")
            )
            if cached_project:
                return normalize_project_name(str(cached_project))
            project_match = PROJECT_RE.search(str(turn.get("user_query", "")))
            if project_match:
                return normalize_project_name(project_match.group(1))
        return ""


class LLMQueryAnalyzer:
    """Uses a small LLM to build QueryPlan, with rule-based fallback for resilience."""

    VALID_INTENTS = {
        "direct_answer",
        "session_context_answer",
        "recent_context_lookup",
        "working_document_lookup",
        "official_knowledge_lookup",
        "organization_memory_lookup",
        "memory_comparison",
    }

    RESPONSE_FORMAT = {
        "type": "json_schema",
        "json_schema": {
            "name": "query_plan",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "intent",
                    "can_answer_directly",
                    "memory_needed",
                    "answer_source",
                    "use_session_context",
                    "memory_weights",
                    "query_rewrites",
                    "filters",
                    "reason",
                ],
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": sorted(VALID_INTENTS),
                    },
                    "can_answer_directly": {"type": "boolean"},
                    "memory_needed": {"type": "boolean"},
                    "answer_source": {
                        "type": "string",
                        "enum": ["direct_answer", "session_only", "memory_prefetch"],
                    },
                    "use_session_context": {"type": "boolean"},
                    "memory_weights": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["stm", "mtm", "ltm"],
                        "properties": {
                            "stm": {"type": "number"},
                            "mtm": {"type": "number"},
                            "ltm": {"type": "number"},
                        },
                    },
                    "query_rewrites": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    # strict json_schema는 모든 object에 additionalProperties: false와
                    # 전체 속성의 required 명시를 요구한다. 선택 항목은 null을 허용해 표현한다.
                    # 자유형 object로 두면 provider가 400을 돌려준다.
                    #
                    # 값 목록은 build_response_format()이 코퍼스에서 enum으로 채운다.
                    # enum이 없으면 LLM이 "공식 계획서" 같은 자연어를 넣고,
                    # 하드 필터에 걸려 검색 결과가 0건이 된다.
                    "filters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["project", "document_types"],
                        "properties": {
                            "project": {"type": ["string", "null"]},
                            "document_types": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "reason": {"type": "string"},
                },
            },
        },
    }

    def __init__(
        self,
        client: Any,
        model: str,
        fallback: RuleBasedQueryAnalyzer | None = None,
        max_tokens: int = 4096,
        vocabulary: dict[str, list[str]] | None = None,
    ):
        self.client = client
        self.model = model
        self.fallback = fallback or RuleBasedQueryAnalyzer()
        self.max_tokens = max_tokens
        self.vocabulary = vocabulary or {}
        self.response_format = self.build_response_format(self.vocabulary)
        self.last_usage: dict[str, Any] = {}
        self.last_fallback_used: bool = False

    @classmethod
    def build_response_format(
        cls,
        vocabulary: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """코퍼스 어휘를 enum으로 박아 넣은 response_format을 만든다.

        vocabulary가 없으면 기본 스키마를 그대로 쓴다.
        enum에는 null을 함께 넣어야 nullable과 함께 쓸 수 있다.
        """
        schema = copy.deepcopy(cls.RESPONSE_FORMAT)
        if not vocabulary:
            return schema
        filters = schema["json_schema"]["schema"]["properties"]["filters"]["properties"]
        projects = list(vocabulary.get("project") or [])
        if projects:
            filters["project"]["enum"] = [*projects, None]
        source_types = list(vocabulary.get("source_type") or [])
        if source_types:
            filters["document_types"]["items"]["enum"] = source_types
        return schema

    def analyze(self, query: str, recent_turns: list[dict[str, Any]] | None = None) -> QueryPlan:
        recent_turns = recent_turns or []
        self.last_usage = {}
        self.last_fallback_used = False
        fallback_plan = self.fallback.analyze(query, recent_turns)
        try:
            raw_plan = self._call_llm(query, recent_turns)
            return self._normalize_plan(raw_plan, query, fallback_plan)
        except Exception as exc:
            # 평가할 때 이 턴은 LLM analyzer 결과가 아니므로 분리해서 봐야 한다.
            self.last_fallback_used = True
            return QueryPlan(
                **{
                    **fallback_plan.to_dict(),
                    "reason": f"{fallback_plan.reason} (LLM query analyzer fallback: {exc})",
                }
            )

    def _call_llm(
        self,
        query: str,
        recent_turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        compact_turns = [
            {
                "user_query": turn.get("user_query", ""),
                "query_intent": turn.get("query_intent", ""),
                "project": turn.get("query_analysis", {}).get("filters", {}).get("project", ""),
                "source_ids": turn.get("source_ids", []),
            }
            for turn in recent_turns[-4:]
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 조직지식 에이전트의 Query Analyzer다. "
                    "문서 내용의 실제 존재 여부를 판단하지 말고, 사용자의 질문 의도와 "
                    "검색 전략만 결정한다. 반드시 JSON schema에 맞춰 답한다. "
                    "모든 문자열 값은 한국어로 작성하고, intent는 허용 목록 중 하나만 사용한다."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_query": query,
                        "recent_session_turns": compact_turns,
                        "allowed_intents": sorted(self.VALID_INTENTS),
                        "memory_tiers": {
                            "stm": "최신 대화, 오늘/방금/아까 결정, 세션성 정보",
                            "mtm": "최근 회의록, 제안서 초안, 진행 중 보고서",
                            "ltm": "공식 계획서, 승인 문서, 조직 기준",
                        },
                        "rules": [
                            "일반 개념 질문이면 direct_answer와 memory_needed=false",
                            "아까 말한거 요약/정리처럼 이전 대화만 묻는 질문이면 session_context_answer, answer_source=session_only, memory_needed=false",
                            "이전 대화를 참고하되 일정/담당자/근거/문서/공식 기준 확인이 필요하면 memory_needed=true",
                            "최근/아까/오늘/후속 질문이면 STM 비중을 높임",
                            "회의록/초안/보고서 질문이면 MTM 비중을 높임",
                            "공식/최종/기준/정책 질문이면 LTM 비중을 높임",
                            "비교/충돌/차이 질문이면 관련 tier를 함께 검색",
                            "query_rewrites에는 원 질문을 첫 항목으로 포함",
                            "query_rewrites에 공식/최종/최신 같은 tier 신호어를 덧붙이지 않는다",
                            "filters.project가 있으면 표준 띄어쓰기 형태로 포함",
                        ],
                        # 규칙 기반 결과를 프롬프트에 넣지 않는다.
                        # 완성된 정답을 참고자료로 주면 작은 모델은 그대로 복사하고,
                        # analyzer는 비용만 쓰는 메아리가 된다.
                        # fallback은 _normalize_plan의 안전망으로만 쓴다.
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.client.chat(
            messages,
            [],
            model=self.model,
            temperature=0.0,
            response_format=self.response_format,
            # 상한이 없으면 작은 모델이 상한까지 토큰을 뱉는 경우가 있다.
            # 잘려서 JSON 파싱이 실패하면 규칙 기반 fallback으로 안전하게 넘어간다.
            max_tokens=self.max_tokens,
        )
        message = response.get("message") or {}
        self.last_usage = response.get("usage") or {}
        content = str(message.get("content") or "").strip()
        return self._loads_json_object(content)

    def _loads_json_object(self, content: str) -> dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(content[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("query analyzer response is not a JSON object")
        return data

    def _normalize_plan(
        self,
        raw: dict[str, Any],
        query: str,
        fallback_plan: QueryPlan,
    ) -> QueryPlan:
        intent = str(raw.get("intent") or fallback_plan.intent)
        if intent not in self.VALID_INTENTS:
            intent = fallback_plan.intent

        memory_needed = self._as_bool(raw.get("memory_needed"), fallback_plan.memory_needed)
        can_answer_directly = self._as_bool(
            raw.get("can_answer_directly"),
            fallback_plan.can_answer_directly,
        )
        answer_source = str(raw.get("answer_source") or fallback_plan.answer_source)
        if answer_source not in {"direct_answer", "session_only", "memory_prefetch"}:
            answer_source = fallback_plan.answer_source
        if answer_source == "session_only":
            memory_needed = False
            can_answer_directly = True
        elif answer_source == "memory_prefetch":
            memory_needed = True
        # LLM 오판 방어.
        # 규칙 기반은 검색이 필요하다고 보는데 LLM만 불필요라고 하면 검색을 유지한다.
        # 불필요한 검색은 토큰을 더 쓸 뿐이지만, 필요한 검색을 건너뛰면
        # 근거 없이 "정보가 없다"고 답하게 되어 실패 방향이 훨씬 나쁘다.
        if fallback_plan.memory_needed and not memory_needed:
            memory_needed = True
            answer_source = "memory_prefetch"
            can_answer_directly = False
        use_session_context = self._as_bool(
            raw.get("use_session_context"),
            fallback_plan.use_session_context,
        )
        weights = self._normalize_weights(raw.get("memory_weights"), memory_needed, fallback_plan)
        rewrites = [
            str(item).strip()
            for item in raw.get("query_rewrites", [])
            if str(item).strip()
        ]
        if query.strip() not in rewrites:
            rewrites.insert(0, query.strip())
        for rewrite in fallback_plan.query_rewrites:
            if rewrite not in rewrites:
                rewrites.append(rewrite)
        filters = self._safe_filters(raw.get("filters"), fallback_plan)

        return QueryPlan(
            intent=intent,
            can_answer_directly=can_answer_directly,
            memory_needed=memory_needed,
            answer_source=answer_source,
            use_session_context=use_session_context,
            memory_weights=weights,
            query_rewrites=list(dict.fromkeys(rewrites)),
            filters=filters,
            reason=str(raw.get("reason") or fallback_plan.reason),
        )

    #: MemoryStore._filter_weight가 실제로 해석하는 키만 통과시킨다.
    ALLOWED_FILTER_KEYS = {
        "project",
        "source_type",
        "status",
        "document_types",
        "date_range",
    }

    def _safe_filters(
        self,
        raw_filters: Any,
        fallback_plan: QueryPlan,
    ) -> dict[str, Any]:
        source = raw_filters if isinstance(raw_filters, dict) else {}
        filters = {
            key: value
            for key, value in source.items()
            if key in self.ALLOWED_FILTER_KEYS and value
        }
        fallback_project = fallback_plan.filters.get("project")
        if fallback_project:
            # 규칙 기반 추출은 정규식과 표기 정규화라 결정적이다.
            # 작은 모델이 과제명을 영어로 번역하거나 임의로 바꾸는 경우가 있는데,
            # project는 하드 필터라 값이 틀어지면 후보가 0건이 된다.
            filters["project"] = fallback_project
        elif filters.get("project"):
            filters["project"] = normalize_project_name(str(filters["project"]))
        return filters

    def _as_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1", "needed"}:
                return True
            if lowered in {"false", "no", "n", "0", "none"}:
                return False
        return default

    def _normalize_weights(
        self,
        raw: Any,
        memory_needed: bool,
        fallback_plan: QueryPlan,
    ) -> dict[str, float]:
        if not memory_needed:
            return {"stm": 0.0, "mtm": 0.0, "ltm": 0.0}
        if not isinstance(raw, dict):
            return fallback_plan.memory_weights
        weights = {
            tier: max(0.0, min(1.0, float(raw.get(tier, 0.0) or 0.0)))
            for tier in ("stm", "mtm", "ltm")
        }
        total = sum(weights.values())
        if total <= 0:
            return fallback_plan.memory_weights
        return {tier: round(value / total, 4) for tier, value in weights.items()}
