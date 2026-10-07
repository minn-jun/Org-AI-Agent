"""세션 전달 원장의 전달량 효과 측정 — 유료 호출 없음.

연속 질문 세 턴을 `SESSION_EVIDENCE_LEDGER` 켜고/끄고 돌려, 모델이 받는
근거 블록의 크기와 **턴 간 재전송 문서 수**를 센다.

왜 이 스크립트가 필요한가
-------------------------
2026-10-06 기준 실제 호출 세션 로그 201개가 **전부 1턴**이다. 연속 질문에서
같은 문서가 다시 실리는지는 측정된 적이 없다. 유료 호출로 재려면 턴마다
답변을 받아야 하므로, 먼저 무료로 전달량만 본다.

읽는 법
-------
LLM은 MockLLMClient다. 답변 품질은 보지 않는다. 보는 것은 전달량뿐이다.
LTM은 실제 과제 청크 코퍼스이고 STM · MTM은 합성 시드다(종료 과제라 현재
자료가 없다). 그래서 이 숫자는 **전달량 비교용이고 성능이 아니다.**

실행
----
    PYTHONIOENCODING=utf-8 LTM_CORPUS=auto python scripts/measure_session_ledger.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LTM_CORPUS", "auto")

from org_agent_mvp.agent_runtime import AgentRuntime  # noqa: E402
from org_agent_mvp.config import AppConfig  # noqa: E402
from org_agent_mvp.memory_store import build_memory_store  # noqa: E402
from org_agent_mvp.mock_llm import MockLLMClient  # noqa: E402
from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer  # noqa: E402


#: 후속 질문의 밀착도를 셋으로 나눠 본다. 원장의 이득은 밀착도에 비례한다.
SCENARIOS: dict[str, list[str]] = {
    "서로 다른 주제": [
        "20200504 과제의 전체 연구개발기간은 언제부터 언제까지야?",
        "그럼 이 과제는 지금까지 어디까지 진행되었어?",
        "예산 집행 현황은 어떻게 돼?",
    ],
    "같은 대상을 파고드는 후속": [
        "20200504 과제의 전체 연구개발기간은 언제부터 언제까지야?",
        "그 연구개발기간이 적힌 문서가 뭐야?",
        "그 문서에서 1차년도 연구개발기간은 어떻게 나와 있어?",
    ],
    "거의 같은 질문 반복": [
        "20200504 과제의 전체 연구개발기간은 언제부터 언제까지야?",
        "연구개발기간 다시 알려줘",
        "연구개발기간이 언제라고 했지?",
    ],
}

COLUMNS = (
    ("turn", "턴", 3),
    ("candidates", "후보", 5),
    ("bodies", "본문", 5),
    ("references", "참조", 5),
    ("evidence_chars", "근거글자", 10),
    ("body_chars", "본문글자", 10),
    ("context_chars", "시스템글자", 11),
    ("resent", "재전송", 7),
    ("tool_calls", "도구", 5),
)

TOTALS = (
    ("evidence_chars", "근거 블록 글자"),
    ("body_chars", "본문 글자"),
    ("context_chars", "시스템 메시지 글자"),
    ("resent", "재전송 문서(턴 간)"),
)


def evidence_block(result: dict) -> list[dict]:
    """프롬프트에 실제로 실린 근거 카드 목록을 되읽는다."""
    for message in result["messages"]:
        content = str(message.get("content", ""))
        if content.startswith("[RUNTIME_CONTEXT]"):
            body = content.split("[RUNTIME_CONTEXT]\n", 1)[1]
            payload = json.loads(body.split("\n[/RUNTIME_CONTEXT]", 1)[0])
            return payload.get("prefetched_evidence", [])
    return []


def body_chars(cards: list[dict]) -> int:
    return sum(
        len(str(card.get("content_excerpt") or ""))
        + sum(
            len(str(part.get("content") or ""))
            for part in (card.get("additional_excerpts") or [])
        )
        for card in cards
    )


def run(turns: list[str], *, ledger: bool) -> list[dict]:
    base = AppConfig.load()
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        config = replace(
            base, project_root=Path(temp_dir), session_evidence_ledger=ledger
        )
        runtime = AgentRuntime(
            config=config,
            client=MockLLMClient(),
            memory_store=build_memory_store(config),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )
        session_id: str | None = None
        seen: set[str] = set()
        for index, question in enumerate(turns, start=1):
            result = runtime.run(question, session_id=session_id)
            session_id = result["session_id"]
            trace = result["trace"]
            cards = evidence_block(result)
            injected = set(trace["injected_sources"])
            rows.append(
                {
                    "turn": index,
                    "candidates": trace["prefetch"]["result_count"],
                    "bodies": sum(1 for c in cards if c.get("content_excerpt")),
                    "references": len(trace["reference_sources"]),
                    "evidence_chars": len(json.dumps(cards, ensure_ascii=False)),
                    "body_chars": body_chars(cards),
                    "context_chars": sum(
                        len(str(message.get("content", "")))
                        for message in result["messages"]
                        if message.get("role") == "system"
                    ),
                    # 앞선 턴에 이미 본문을 전달한 문서를 또 전달한 건수.
                    "resent": len(injected & seen),
                    "tool_calls": len(trace["tool_calls"]),
                }
            )
            seen |= injected | set(trace["tool_sources"])
    return rows


def report(off: list[dict], on: list[dict]) -> None:
    header = "".join(f"{label:>{width}}" for _, label, width in COLUMNS)
    for name, rows in (("원장 OFF (현재 기본값)", off), ("원장 ON", on)):
        print(f"── {name}")
        print(header)
        for row in rows:
            print("".join(f"{row[key]:>{width}}" for key, _, width in COLUMNS))
        print()
    for key, label in TOTALS:
        a = sum(row[key] for row in off)
        b = sum(row[key] for row in on)
        delta = f"{(b - a) / a * 100:+.1f}%" if a else "—"
        print(f"{label:<20} OFF {a:>8}   ON {b:>8}   {delta}")


def main() -> None:
    print(f"LTM corpus: {AppConfig.load().ltm_corpus_path}")
    for name, turns in SCENARIOS.items():
        print()
        print("=" * 72)
        print(f"시나리오: {name}")
        for index, question in enumerate(turns, start=1):
            print(f"  {index}. {question}")
        print()
        report(run(turns, ledger=False), run(turns, ledger=True))


if __name__ == "__main__":
    main()
