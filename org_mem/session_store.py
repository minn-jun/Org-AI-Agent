from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


#: 세션 파일에서 전달 원장이 들어가는 칸. `turns`와 나란히 둔다.
DELIVERED_KEY = "delivered_evidence"


def ledger_card(record: dict[str, Any]) -> dict[str, Any]:
    """원장 기록을 근거 카드 모양으로 되돌린다.

    `expand_evidence`가 보는 필드(`source_ref.document_id` · `chunk_index`)만
    맞춰 주면 원문은 청크 코퍼스에서 다시 읽는다. 본문을 원장에 쌓지 않는
    이유가 그것이다 — 턴마다 세션 파일이 불어난다. `excerpt`는 코퍼스에서
    원문을 못 찾을 때(STM · MTM 카드)만 쓰는 대비책이다.
    """
    return {
        "evidence_id": record.get("evidence_id"),
        "tier": record.get("tier"),
        "title": record.get("title"),
        "date": record.get("date"),
        "project": record.get("project"),
        "summary": record.get("summary"),
        "content_excerpt": record.get("excerpt", ""),
        "source_ref": {
            "document_id": record.get("document_id"),
            "chunk_index": record.get("chunk_index"),
            "page_nos": list(record.get("page_nos") or []),
        },
        "delivered_in_turn": record.get("turn_index"),
    }


class SessionStore:
    """세션 기록 저장소.

    세션 파일은 **모든 턴을 계속 누적한다.** 오래된 턴을 지우지 않는다.
    이전에는 최근 N턴만 남기고 잘라내서, 그 이전 대화가 복구 불가능하게 사라졌다.

    컨텍스트에 몇 턴을 넣을지는 저장과 별개 문제이며,
    `AgentRuntime`이 읽는 시점에 자른다.
    """

    def __init__(self, root: Path, cache_turns: int = 8):
        self.root = root
        # 저장 상한이 아니라, 호출부가 컨텍스트용으로 참고하는 기본값이다.
        self.cache_turns = cache_turns

    def create(self) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        session = {
            "session_id": datetime.now().strftime("session-%Y%m%d-%H%M%S-") + uuid4().hex[:6],
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
        self.save(session)
        return session

    def load_or_create(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            path = self.root / f"{session_id}.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        return self.create()

    def append_turn(self, session: dict[str, Any], turn: dict[str, Any]) -> Path:
        turns = list(session.get("turns", []))
        turns.append(turn)
        # 잘라내지 않는다. 전체 기록을 남긴다.
        session["turns"] = turns
        session["turn_count"] = len(turns)
        session["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return self.save(session)

    # ------------------------------------------------------------ 전달 원장
    def delivered(self, session: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """document_id -> 전달 기록. 원장이 없으면 빈 dict다."""
        raw = session.get(DELIVERED_KEY)
        return dict(raw) if isinstance(raw, dict) else {}

    def delivered_cards(self, session: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """evidence_id -> 카드. `expand_evidence` 색인에 그대로 넣을 수 있다."""
        cards: dict[str, dict[str, Any]] = {}
        for record in self.delivered(session).values():
            evidence_id = str(record.get("evidence_id") or "")
            if evidence_id:
                cards[evidence_id] = ledger_card(record)
        return cards

    def record_delivered(
        self,
        session: dict[str, Any],
        cards: list[dict[str, Any]],
        *,
        turn_index: int,
        turn_id: str = "",
        max_docs: int = 200,
        excerpt_chars: int = 600,
    ) -> dict[str, dict[str, Any]]:
        """이번 턴에 **본문을 전달한** 카드를 원장에 적는다.

        같은 문서를 다시 전달해도 **첫 기록을 유지한다.** 언제 처음 줬는지가
        모델에게 주는 정보이고, 두 번째 전달은 원장이 켜져 있으면 애초에
        일어나지 않는다.

        저장은 하지 않는다. 호출부가 `append_turn`을 부르면 함께 저장된다.
        """
        ledger = self.delivered(session)
        for card in cards:
            ref = card.get("source_ref") or {}
            document_id = str(ref.get("document_id") or "")
            if not document_id or document_id in ledger:
                continue
            ledger[document_id] = {
                "document_id": document_id,
                "evidence_id": str(card.get("evidence_id") or ""),
                "tier": card.get("tier"),
                "title": card.get("title"),
                "date": card.get("date"),
                "project": card.get("project"),
                "summary": card.get("summary"),
                "chunk_index": ref.get("chunk_index"),
                "page_nos": list(ref.get("page_nos") or []),
                "excerpt": str(card.get("content_excerpt") or "")[:excerpt_chars],
                "turn_index": turn_index,
                "turn_id": turn_id,
            }
        if max_docs > 0 and len(ledger) > max_docs:
            # 오래된 턴의 것부터 버린다. 최근에 본 문서가 다시 나올 가능성이 높다.
            ordered = sorted(
                ledger.items(), key=lambda item: int(item[1].get("turn_index") or 0)
            )
            ledger = dict(ordered[-max_docs:])
        session[DELIVERED_KEY] = ledger
        return ledger

    def recent(self, session: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
        """컨텍스트에 넘길 최근 턴만 잘라 준다. 저장본은 건드리지 않는다."""
        turns = list(session.get("turns", []))
        window = self.cache_turns if limit is None else limit
        return turns[-window:] if window > 0 else turns

    def save(self, session: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{session['session_id']}.json"
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
