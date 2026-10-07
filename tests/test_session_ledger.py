"""세션 전달 원장 — 턴 간 중복 전달 테스트 (2026-10-06).

지금까지의 중복 제거는 **한 턴 안**에서만 돌았다. `seen_tier_queries`와
`AgentTrace`가 턴마다 새로 만들어지기 때문이다. 그래서 연속 질문에서
같은 문서의 본문이 턴마다 다시 실렸다.

실제 호출 로그 201개가 전부 1턴이라, 이 상황은 측정된 적이 없다.
여기서 모의 실행으로 동작을 고정한다.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from org_agent_mvp.agent_runtime import AgentRuntime, AgentTrace
from org_agent_mvp.config import AppConfig
from org_agent_mvp.memory_store import MemoryStore
from org_agent_mvp.mock_llm import MockLLMClient
from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer
from org_agent_mvp.session_store import DELIVERED_KEY, SessionStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "memory"

#: 사용자가 말한 연속 질문 모양. 같은 과제를 두 번 묻는다.
FIRST = "A 과제 최종 제출일이 언제야?"
SECOND = "그럼 A 과제는 지금까지 어디까지 진행되었어?"


def runtime_for(temp_dir: str, **overrides) -> AgentRuntime:
    overrides.setdefault("session_evidence_ledger", True)
    config = replace(
        AppConfig.load(),
        project_root=Path(temp_dir),
        memory_root=MEMORY_FIXTURE,
        **overrides,
    )
    return AgentRuntime(
        config=config,
        client=MockLLMClient(),
        memory_store=MemoryStore(config.memory_root),
        query_analyzer=RuleBasedQueryAnalyzer(),
    )


def runtime_context(result: dict) -> dict:
    """프롬프트에 실린 RUNTIME_CONTEXT를 되읽는다."""
    for message in result["messages"]:
        content = str(message.get("content", ""))
        if content.startswith("[RUNTIME_CONTEXT]"):
            body = content.split("[RUNTIME_CONTEXT]\n", 1)[1]
            return json.loads(body.split("\n[/RUNTIME_CONTEXT]", 1)[0])
    raise AssertionError("RUNTIME_CONTEXT를 찾지 못했다")


def load_session(result: dict) -> dict:
    return json.loads(Path(result["session_log_path"]).read_text(encoding="utf-8"))


# ------------------------------------------------------- 1. 기본값은 예전 동작


class LedgerDefaultTests(unittest.TestCase):
    def test_ledger_is_off_by_default(self) -> None:
        self.assertFalse(AppConfig.load().session_evidence_ledger)

    def test_off_keeps_sending_the_same_bodies(self) -> None:
        """꺼져 있으면 2턴째도 같은 본문이 그대로 다시 실린다.

        이것이 지금의 동작이고, 원장을 켜는 이유다.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir, session_evidence_ledger=False)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])

            self.assertEqual(second["trace"]["reference_sources"], [])
            self.assertEqual(second["trace"]["session_delivered_sources"], [])
            self.assertNotIn(DELIVERED_KEY, load_session(second))

            # 두 턴의 근거가 겹치는데, 2턴째 카드에도 본문이 들어 있다.
            repeated = set(first["trace"]["injected_sources"]) & set(
                second["trace"]["injected_sources"]
            )
            self.assertTrue(repeated, "겹치는 문서가 없으면 이 테스트는 의미가 없다")
            cards = runtime_context(second)["prefetched_evidence"]
            bodies = [
                card
                for card in cards
                if card.get("source_id") in repeated and card.get("content_excerpt")
            ]
            self.assertTrue(bodies)


# ------------------------------------------------- 2. 켜면 참조 카드로 바뀐다


class LedgerOnTests(unittest.TestCase):
    def test_second_turn_sends_reference_instead_of_body(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])

            delivered = set(first["trace"]["injected_sources"])
            self.assertTrue(delivered)
            self.assertEqual(
                set(second["trace"]["session_delivered_sources"]) & delivered,
                delivered,
            )
            self.assertTrue(second["trace"]["reference_sources"])

            for card in runtime_context(second)["prefetched_evidence"]:
                if card.get("source_id") in second["trace"]["reference_sources"]:
                    # 제목과 출처는 남기고 본문만 뺀다.
                    self.assertNotIn("content_excerpt", card)
                    self.assertNotIn("additional_excerpts", card)
                    self.assertTrue(card.get("body_available"))
                    self.assertEqual(card.get("delivered_in_turn"), 1)
                    self.assertTrue(card.get("title"))

    def test_reference_sources_are_not_counted_as_injected(self) -> None:
        """참조만 준 문서를 주입으로 적으면 "본문을 전달했다"로 읽힌다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])
            trace = second["trace"]
            self.assertFalse(
                set(trace["reference_sources"]) & set(trace["injected_sources"])
            )
            self.assertFalse(
                set(trace["reference_sources"]) & set(trace["final_sources"])
            )

    def test_second_turn_context_gets_shorter(self) -> None:
        """같은 2턴을 켜고/끄고 돌려 컨텍스트 길이를 비교한다."""
        lengths = {}
        for flag in (False, True):
            with tempfile.TemporaryDirectory() as temp_dir:
                runtime = runtime_for(temp_dir, session_evidence_ledger=flag)
                first = runtime.run(FIRST)
                second = runtime.run(SECOND, session_id=first["session_id"])
                lengths[flag] = len(
                    json.dumps(
                        runtime_context(second)["prefetched_evidence"],
                        ensure_ascii=False,
                    )
                )
        self.assertLess(lengths[True], lengths[False])

    def test_guidance_explains_how_to_get_the_body_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])
            context = next(
                str(message.get("content", ""))
                for message in second["messages"]
                if str(message.get("content", "")).startswith("[RUNTIME_CONTEXT]")
            )
            self.assertIn("delivered_in_turn", context.split("[/RUNTIME_CONTEXT]")[1])
            self.assertIn("expand_evidence", context.split("[/RUNTIME_CONTEXT]")[1])


# ------------------------------------------------------ 3. 되불러오는 경로


class LedgerRecallTests(unittest.TestCase):
    def test_expand_tool_is_offered_even_in_full_mode(self) -> None:
        """참조만 보여 주고 본문 경로를 막으면 "근거 없음"으로 답하게 된다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            names = {
                tool["function"]["name"]
                for tool in runtime_for(temp_dir, context_mode="full").tools
            }
            self.assertIn("expand_evidence", names)

            off = {
                tool["function"]["name"]
                for tool in runtime_for(
                    temp_dir, context_mode="full", session_evidence_ledger=False
                ).tools
            }
            self.assertNotIn("expand_evidence", off)

    def test_previous_turn_evidence_id_still_expands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir)
            first = runtime.run(FIRST)
            session = load_session(first)
            index = runtime.session_store.delivered_cards(session)
            self.assertTrue(index)

            evidence_id = next(iter(index))
            message = runtime._execute_expand_evidence(
                {
                    "id": "call_1",
                    "function": {
                        "name": "expand_evidence",
                        "arguments": json.dumps(
                            {"evidence_ids": [evidence_id], "reason": "이전 턴 근거"},
                            ensure_ascii=False,
                        ),
                    },
                },
                index,
                AgentTrace(),
            )
            payload = json.loads(message["content"])
            self.assertEqual(payload["unknown_evidence_ids"], [])
            self.assertTrue(payload["expanded"][0]["content"])


# ------------------------------------------- 4. 도구 결과도 턴 간 중복을 본다


class LedgerToolResultTests(unittest.TestCase):
    def _result(self, doc_ids: list[str]) -> dict:
        return {
            "results": [
                {
                    "evidence_id": f"ev-{doc}",
                    "title": doc,
                    "content_excerpt": "본문" * 10,
                    "source_ref": {"document_id": doc},
                }
                for doc in doc_ids
            ]
        }

    def test_documents_from_earlier_turns_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir, tool_result_dedupe=True)
            trace = AgentTrace()
            trace.session_delivered_sources = ["doc-a"]
            trimmed = runtime._trim_tool_result(
                self._result(["doc-a", "doc-b"]), trace
            )
            self.assertEqual(trimmed["duplicates_removed"], 1)
            self.assertEqual(
                [card["source_ref"]["document_id"] for card in trimmed["results"]],
                ["doc-b"],
            )

    def test_dedupe_off_keeps_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir, tool_result_dedupe=False)
            trace = AgentTrace()
            trace.session_delivered_sources = ["doc-a"]
            trimmed = runtime._trim_tool_result(
                self._result(["doc-a", "doc-b"]), trace
            )
            self.assertEqual(trimmed["duplicates_removed"], 0)
            self.assertEqual(len(trimmed["results"]), 2)


# ------------------------------------------------------------- 5. 원장 자체


class LedgerStoreTests(unittest.TestCase):
    def _card(self, doc: str) -> dict:
        return {
            "evidence_id": f"ev-{doc}",
            "tier": "ltm",
            "title": f"{doc} 제목",
            "date": "2026-06-03",
            "project": "A 과제",
            "summary": "요약",
            "content_excerpt": "본문" * 500,
            "source_ref": {"document_id": doc, "chunk_index": 3, "page_nos": [7]},
        }

    def test_first_delivery_turn_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            session = store.create()
            store.record_delivered(session, [self._card("doc-a")], turn_index=1)
            store.record_delivered(session, [self._card("doc-a")], turn_index=4)
            self.assertEqual(store.delivered(session)["doc-a"]["turn_index"], 1)

    def test_excerpt_is_trimmed_so_the_session_file_stays_small(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            session = store.create()
            store.record_delivered(
                session, [self._card("doc-a")], turn_index=1, excerpt_chars=100
            )
            self.assertEqual(len(store.delivered(session)["doc-a"]["excerpt"]), 100)

    def test_oldest_documents_are_dropped_at_the_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            session = store.create()
            for turn in range(1, 6):
                store.record_delivered(
                    session, [self._card(f"doc-{turn}")], turn_index=turn, max_docs=3
                )
            self.assertEqual(
                sorted(store.delivered(session)), ["doc-3", "doc-4", "doc-5"]
            )

    def test_cards_without_a_document_id_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(Path(temp_dir))
            session = store.create()
            store.record_delivered(session, [{"evidence_id": "ev-x"}], turn_index=1)
            self.assertEqual(store.delivered(session), {})


# ----------------------------------------- 6. 사용자가 말한 연속 질문 시나리오


class FollowUpScenarioTests(unittest.TestCase):
    """"A 과제 종료 기간?" 다음에 "어디까지 진행됐어?"를 묻는 경우."""

    def test_follow_up_turns_the_first_turn_documents_into_references(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])

            trace = second["trace"]
            # ① 1턴에서 본 문서는 대부분 참조로 바뀐다
            self.assertTrue(trace["reference_sources"])
            self.assertLess(
                len(trace["injected_sources"]),
                len(first["trace"]["injected_sources"]),
            )
            # ② 그래도 본문이 0장이 되지는 않는다 (아래 안전장치 테스트 참고)
            self.assertTrue(trace["injected_sources"])
            # ③ 같은 세션이고 과제 추론도 유지된다
            self.assertEqual(first["session_id"], second["session_id"])
            self.assertEqual(trace["query_analysis"]["filters"]["project"], "A 과제")
            # ④ 원장은 두 턴의 문서를 모두 들고 있다
            ledger = load_session(second)[DELIVERED_KEY]
            self.assertTrue(set(first["trace"]["injected_sources"]) <= set(ledger))

    def test_freed_slots_go_to_documents_not_seen_yet(self) -> None:
        """참조로 줄어든 자리를 **새 문서**가 채운다.

        후보 상한을 새 문서 기준으로 세지 않으면, 이미 본 문서가 자리를
        차지해서 2턴째가 1턴째보다 정보가 적어진다.

        고정 fixture(A 과제 문서 7건)에서 상한이 8이면 2턴에 새로 볼 것이
        남지 않으므로, 여기서는 상한을 2로 낮춰 여유가 있는 상황을 만든다.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir, prefetch_top_k=2)
            first = runtime.run(FIRST)
            # 2턴을 돌리기 **전에** 읽어야 한다. 세션 파일은 같은 경로라
            # 나중에 읽으면 2턴의 문서까지 원장에 들어 있다.
            seen = set(load_session(first)[DELIVERED_KEY])
            second = runtime.run(SECOND, session_id=first["session_id"])

            fresh = [
                source
                for source in second["trace"]["injected_sources"]
                if source not in seen
            ]
            self.assertTrue(fresh, "새 문서가 하나도 안 들어오면 참조 전환 이득이 없다")
            # 참조는 상한을 넘겨 따라붙는다 — 본문 2장 자리를 뺏지 않는다.
            self.assertGreater(second["trace"]["prefetch"]["result_count"], 2)

    def test_a_turn_never_ends_up_with_zero_bodies(self) -> None:
        """새로 볼 문서가 없어도 본문 몇 장은 다시 보낸다.

        이전 턴의 본문은 messages에 남지 않는다(턴마다 새로 만든다).
        전부 참조로 바꾸면 모델이 받는 것은 제목과 답변 요약 500자뿐이고,
        "근거 없음"으로 답하거나 expand_evidence로 왕복을 늘린다.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir, session_ledger_min_bodies=2)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])
            # fixture의 A 과제 문서는 1턴에서 이미 다 나갔다.
            self.assertEqual(len(second["trace"]["injected_sources"]), 2)
            self.assertTrue(second["trace"]["reference_sources"])

    def test_min_bodies_zero_allows_an_all_reference_turn(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir, session_ledger_min_bodies=0)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])
            self.assertEqual(second["trace"]["injected_sources"], [])

    def test_third_turn_resends_at_most_the_minimum(self) -> None:
        """턴이 쌓여도 다시 보내는 본문은 안전장치 몫(기본 2장)을 넘지 않는다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = runtime_for(temp_dir)
            first = runtime.run(FIRST)
            second = runtime.run(SECOND, session_id=first["session_id"])
            seen = set(load_session(second)[DELIVERED_KEY])
            third = runtime.run(
                "A 과제 예산 검토는 어떻게 됐어?", session_id=first["session_id"]
            )

            resent = seen & set(third["trace"]["injected_sources"])
            self.assertLessEqual(
                len(resent), runtime.config.session_ledger_min_bodies
            )
            # 나머지는 참조로만 들어간다.
            references = set(third["trace"]["reference_sources"])
            self.assertTrue(references)
            for card in runtime_context(third)["prefetched_evidence"]:
                if card.get("source_id") in references:
                    self.assertNotIn("content_excerpt", card)


if __name__ == "__main__":
    unittest.main()
