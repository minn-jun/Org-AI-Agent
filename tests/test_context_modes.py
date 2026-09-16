from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from org_agent_mvp.agent_runtime import AgentRuntime
from org_agent_mvp.config import AppConfig
from org_agent_mvp.context_builder import ContextBuilder
from org_agent_mvp.memory_store import MemoryStore
from org_agent_mvp.mock_llm import MockLLMClient
from org_agent_mvp.prefetch import MemoryPrefetcher
from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer


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
        from org_agent_mvp.prompts import SYSTEM_PROMPT, system_prompt

        previous = os.environ.pop("PROMPT_PREFETCH_HINT", None)
        try:
            self.assertEqual(system_prompt(), SYSTEM_PROMPT)
        finally:
            if previous is not None:
                os.environ["PROMPT_PREFETCH_HINT"] = previous

    def test_prefetch_hint_can_be_switched_on(self) -> None:
        from org_agent_mvp.prompts import SYSTEM_PROMPT, system_prompt

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
        self.assertIn("이미 STM·MTM·LTM을 모두 검색해 고른 결과", text)

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
            from org_agent_mvp.agent_runtime import AgentTrace

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
