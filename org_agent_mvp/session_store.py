from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


class SessionStore:
    def __init__(self, root: Path, cache_turns: int = 8):
        self.root = root
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
        session["turns"] = turns[-self.cache_turns :]
        session["updated_at"] = datetime.now().isoformat(timespec="seconds")
        return self.save(session)

    def save(self, session: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{session['session_id']}.json"
        path.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
