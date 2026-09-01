from __future__ import annotations

from typing import Any


RETRIEVE_MEMORY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve_memory",
        "description": (
            "Search organization memory. Use this whenever the user asks about "
            "recent meetings, schedules, action items, documents, official plans, "
            "company standards, project facts, or source-grounded answers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tier": {
                    "type": "string",
                    "enum": ["stm", "mtm", "ltm", "all"],
                    "description": "Memory tier to search.",
                },
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters.",
                    "properties": {
                        "project": {"type": "string"},
                        "date_range": {"type": "string"},
                        "document_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "source_type": {"type": "string"},
                        "status": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of evidence cards to return.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this retrieval is needed.",
                },
            },
            "required": ["tier", "query", "reason"],
            "additionalProperties": False,
        },
    },
}


#: 이미 제시된 근거의 원문만 꺼내는 도구.
#: retrieve_memory와 달리 새로 검색하지 않고 턴 안에 들고 있는 카드를 조회한다.
EXPAND_EVIDENCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "expand_evidence",
        "description": (
            "Read the full text of evidence cards that were already listed in "
            "the runtime context. This does not run a new search. Use it only "
            "for cards whose summary is not enough to answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 8,
                    "description": "evidence_id values taken from the runtime context.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why the summaries are not sufficient.",
                },
            },
            "required": ["evidence_ids", "reason"],
            "additionalProperties": False,
        },
    },
}

