from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from org_mem.agent_runtime import AgentRuntime
from org_mem.config import AppConfig
from org_mem.context_builder import ContextBuilder
from org_mem.memory_store import MemoryStore
from org_mem.mock_llm import MockLLMClient
from org_mem.prefetch import MemoryPrefetcher
from org_mem.query_analyzer import RuleBasedQueryAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "memory"
QUESTION = "A 과제 예산 산정 기준이 뭐야?"


def build_cards() -> tuple[RuleBasedQueryAnalyzer, list[dict]]:
    store = MemoryStore(MEMORY_FIXTURE)
    plan = RuleBasedQueryAnalyzer().analyze(QUESTION)
    cards = MemoryPrefetcher(store, total_top_k=8).prefetch(plan).cards
    return plan, cards


def parse_context(text: str) -> dict:
    raw = text.split("[RUNTIME_CONTEXT]", 1)[1].split("[/RUNTIME_CONTEXT]", 1)[0]
    return json.loads(raw)


class ContextModeTests(unittest.TestCase):
    def test_full_mode_includes_every_body(self) -> None:
        plan, cards = build_cards()
        payload = parse_context(ContextBuilder(context_mode="full").build(plan, cards, []))
        evidence = payload["prefetched_evidence"]
        self.assertTrue(evidence)
        self.assertTrue(all("content_excerpt" in item for item in evidence))

    def test_summary_mode_omits_every_body(self) -> None:
        plan, cards = build_cards()
        payload = parse_context(ContextBuilder(context_mode="summary").build(plan, cards, []))
        evidence = payload["prefetched_evidence"]
        self.assertTrue(evidence)
        self.assertTrue(all("content_excerpt" not in item for item in evidence))
        self.assertTrue(all(item["body_available"] for item in evidence))

    def test_hybrid_mode_keeps_only_top_bodies(self) -> None:
        plan, cards = build_cards()
        builder = ContextBuilder(context_mode="hybrid", hybrid_full_cards=2)
        evidence = parse_context(builder.build(plan, cards, []))["prefetched_evidence"]
        self.assertGreater(len(evidence), 2)
        self.assertTrue(all("content_excerpt" in item for item in evidence[:2]))
        self.assertTrue(all("content_excerpt" not in item for item in evidence[2:]))

    def test_summary_context_is_smaller_than_full(self) -> None:
        plan, cards = build_cards()
        full = ContextBuilder(context_mode="full").build(plan, cards, [])
        summary = ContextBuilder(context_mode="summary").build(plan, cards, [])
        hybrid = ContextBuilder(context_mode="hybrid").build(plan, cards, [])
        self.assertLess(len(summary), len(hybrid))
        self.assertLess(len(hybrid), len(full))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ContextBuilder(context_mode="nope")


class ExpandEvidenceTests(unittest.TestCase):
    def _runtime(self, context_mode: str, temp_dir: str) -> AgentRuntime:
        config = replace(
            AppConfig.load(),
            project_root=Path(temp_dir),
            memory_root=MEMORY_FIXTURE,
            context_mode=context_mode,
        )
        return AgentRuntime(
            config=config,
            client=MockLLMClient(),
            memory_store=MemoryStore(config.memory_root),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )

    def test_full_mode_does_not_expose_expand_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime("full", temp_dir)
            names = [tool["function"]["name"] for tool in runtime.tools]
            self.assertEqual(names, ["retrieve_memory"])

    def test_prefetch_hint_is_off_by_default(self) -> None:
        """기본 동작은 2026-09-16 이전과 같아야 한다."""
        from org_mem.prompts import SYSTEM_PROMPT, system_prompt

        previous = os.environ.pop("PROMPT_PREFETCH_HINT", None)
        try:
            self.assertEqual(system_prompt(), SYSTEM_PROMPT)
        finally:
            if previous is not None:
                os.environ["PROMPT_PREFETCH_HINT"] = previous

    def test_prefetch_hint_can_be_switched_on(self) -> None:
        from org_mem.prompts import SYSTEM_PROMPT, system_prompt

        previous = os.environ.get("PROMPT_PREFETCH_HINT")
        os.environ["PROMPT_PREFETCH_HINT"] = "1"
        try:
            text = system_prompt()
        finally:
            if previous is None:
                os.environ.pop("PROMPT_PREFETCH_HINT", None)
            else:
                os.environ["PROMPT_PREFETCH_HINT"] = previous
        self.assertIn(SYSTEM_PROMPT.rstrip(), text)
        self.assertIn("미리 검색해 고른 결과", text)
        # 2026-10-19: "세 계층을 모두 검색했다"고 단정하지 않는다. Jev가 계층을
        # 좁히면 거짓이 되고, 좁혀서 생긴 누락을 되살릴 창구를 막는다.
        self.assertNotIn("이미 STM·MTM·LTM을 모두 검색해", text)
        self.assertIn("검색하지 않은 계층이 있다고 적혀 있으면", text)

    def test_evidence_strict_is_off_by_default(self) -> None:
        """기본 동작은 2026-09-28 이전과 같아야 한다."""
        from org_mem.prompts import SYSTEM_PROMPT, system_prompt

        previous = os.environ.pop("PROMPT_EVIDENCE_STRICT", None)
        try:
            self.assertEqual(system_prompt(), SYSTEM_PROMPT)
        finally:
            if previous is not None:
                os.environ["PROMPT_EVIDENCE_STRICT"] = previous

    def test_evidence_strict_can_be_switched_on(self) -> None:
        from org_mem.prompts import SYSTEM_PROMPT, system_prompt

        previous = os.environ.get("PROMPT_EVIDENCE_STRICT")
        os.environ["PROMPT_EVIDENCE_STRICT"] = "1"
        try:
            text = system_prompt()
        finally:
            if previous is None:
                os.environ.pop("PROMPT_EVIDENCE_STRICT", None)
            else:
                os.environ["PROMPT_EVIDENCE_STRICT"] = previous
        self.assertIn(SYSTEM_PROMPT.rstrip(), text)
        self.assertIn("근거 본문에 있는 값만", text)

    def test_summary_mode_exposes_expand_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime("summary", temp_dir)
            names = [tool["function"]["name"] for tool in runtime.tools]
            self.assertIn("expand_evidence", names)

    def test_summary_run_expands_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._runtime("summary", temp_dir).run(QUESTION)
            trace = result["trace"]
            self.assertEqual(trace["context_mode"], "summary")
            self.assertGreater(trace["expansion_count"], 0)
            self.assertTrue(result["answer"])

    def test_full_run_never_expands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._runtime("full", temp_dir).run(QUESTION)
            self.assertEqual(result["trace"]["expansion_count"], 0)

    def test_expand_returns_body_and_flags_unknown_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime("summary", temp_dir)
            _, cards = build_cards()
            index = {str(c["evidence_id"]): c for c in cards}
            known = next(iter(index))
            from org_mem.agent_runtime import AgentTrace

            trace = AgentTrace()
            message = runtime._execute_expand_evidence(
                {
                    "id": "call_1",
                    "function": {
                        "name": "expand_evidence",
                        "arguments": json.dumps(
                            {"evidence_ids": [known, "ev_ltm_없는문서"], "reason": "확인"},
                            ensure_ascii=False,
                        ),
                    },
                },
                index,
                trace,
            )
            payload = json.loads(message["content"])
            self.assertEqual(payload["expanded_count"], 1)
            self.assertTrue(payload["expanded"][0]["content"])
            self.assertEqual(payload["unknown_evidence_ids"], ["ev_ltm_없는문서"])
            self.assertEqual(trace.expanded_ids, [known])


if __name__ == "__main__":
    unittest.main()


class SearchScopeNoticeTests(unittest.TestCase):
    """검색하지 않은 계층을 모델에게 알려 주는지 본다.

    2026-10-16 실측(30턴, Jev): 기대 근거가 전달되지 않은 3턴이 전부 도구 호출
    0회였다. 모델은 들어온 카드만 보고 안 들어온 것을 볼 수 없어서, 부족한 줄
    모른 채 답했다. 이 안내가 그 판단 재료다.
    """

    def _text(self, weights: dict[str, float], *, memory_needed: bool = True) -> str:
        plan, cards = build_cards()
        plan = replace(plan, memory_weights=weights, memory_needed=memory_needed)
        builder = ContextBuilder(context_mode="full")
        return builder.build(plan, cards, recent_turns=[])

    def test_notice_lists_opened_and_skipped_tiers(self) -> None:
        text = self._text({"stm": 0.0, "mtm": 1.0, "ltm": 0.0})
        self.assertIn("mtm만 검색했고", text)
        self.assertIn("stm·ltm는 검색하지 않았다", text)
        # 허용형("부족하면 불러도 된다")이 아니라 의무 절차여야 한다.
        # 2026-10-19 실측: 허용형 문구로는 도구 호출이 한 건도 늘지 않았다.
        self.assertIn("답하기 전에 먼저 판단한다", text)
        self.assertIn("retrieve_memory로 검색하지 않은 계층을 확인한다", text)

    def test_notice_is_absent_when_every_tier_was_searched(self) -> None:
        """규칙 분석기는 세 계층에 모두 0보다 큰 가중치를 준다 — 붙지 않는다."""
        text = self._text({"stm": 0.1, "mtm": 0.2, "ltm": 0.7})
        self.assertNotIn("검색하지 않았다", text)

    def test_notice_is_absent_when_search_is_not_needed(self) -> None:
        text = self._text(
            {"stm": 0.0, "mtm": 0.0, "ltm": 0.0}, memory_needed=False
        )
        self.assertNotIn("검색하지 않았다", text)

    def test_notice_does_not_leak_the_branch_decision(self) -> None:
        """안 본 계층이라는 **사실만** 준다. 분기 이름·확률은 주지 않는다.

        판정 결과를 답변 모델에 흘리면 생성이 판정에 끌려간다.
        """
        text = self._text({"stm": 1.0, "mtm": 0.0, "ltm": 1.0})
        self.assertIn("stm·ltm만 검색했고", text)
        for leaked in ("stm_ltm", "분기", "confidence", "probabilit"):
            self.assertNotIn(leaked, text)

    def test_notice_can_be_switched_off_for_reproduction(self) -> None:
        """예전 측정을 재현하려면 끌 수 있어야 한다."""
        plan, cards = build_cards()
        plan = replace(plan, memory_weights={"stm": 0.0, "mtm": 1.0, "ltm": 0.0})
        builder = ContextBuilder(context_mode="full", search_scope_notice=False)
        text = builder.build(plan, cards, recent_turns=[])
        self.assertNotIn("검색하지 않았다", text)
