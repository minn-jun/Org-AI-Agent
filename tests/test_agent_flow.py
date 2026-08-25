from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from org_agent_mvp.agent_runtime import AgentRuntime
from org_agent_mvp.config import AppConfig
from org_agent_mvp.memory_store import MemoryStore
from org_agent_mvp.mock_llm import MockLLMClient
from org_agent_mvp.prefetch import MemoryPrefetcher
from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QueryAnalyzerTests(unittest.TestCase):
    def test_comparison_uses_all_memory_tiers(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze(
            "A 과제의 최근 일정이 공식 계획과 충돌해?"
        )
        self.assertEqual(plan.intent, "memory_comparison")
        self.assertTrue(plan.memory_needed)
        self.assertGreater(plan.memory_weights["stm"], 0)
        self.assertGreater(plan.memory_weights["mtm"], 0)
        self.assertGreater(plan.memory_weights["ltm"], 0)

    def test_follow_up_resolves_project_from_previous_turn(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze(
            "그 일정의 담당자는?",
            [{"user_query": "A 과제 공식 일정 알려줘"}],
        )
        self.assertEqual(plan.intent, "recent_context_lookup")
        self.assertEqual(plan.filters["project"], "A 과제")
        self.assertIn("A 과제", plan.query_rewrites[-1])


class PrefetchTests(unittest.TestCase):
    def test_prefetch_respects_total_top_k(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A 과제 공식 일정과 최근 결정 비교")
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)
        self.assertEqual(sum(result.allocations.values()), 8)
        self.assertLessEqual(len(result.cards), 8)
        self.assertTrue(result.cards)


class SessionRuntimeTests(unittest.TestCase):
    def test_mock_runtime_continues_a_saved_session(self) -> None:
        base = AppConfig.load()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                base,
                project_root=Path(temp_dir),
                memory_root=PROJECT_ROOT / "memory_seed",
            )
            runtime = AgentRuntime(config, MockLLMClient(), MemoryStore(config.memory_root))
            first = runtime.run("A 과제 최근 일정이 공식 계획과 충돌해?")
            second = runtime.run("그 일정의 담당자와 남은 일은?", session_id=first["session_id"])

            self.assertEqual(first["session_id"], second["session_id"])
            self.assertEqual(second["trace"]["query_analysis"]["filters"]["project"], "A 과제")
            self.assertTrue(Path(second["session_log_path"]).exists())
            self.assertGreaterEqual(second["trace"]["prefetch"]["result_count"], 1)


if __name__ == "__main__":
    unittest.main()
