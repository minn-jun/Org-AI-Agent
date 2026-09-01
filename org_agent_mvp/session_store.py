from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


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
