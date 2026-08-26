from __future__ import annotations

import json
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

    def test_project_name_without_space_is_normalized(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze("A과제 최근 회의 내용 알려줘")

        self.assertEqual(plan.filters["project"], "A 과제")

    def test_project_word_is_not_limited_to_seed_task_names(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze("crm프로젝트 최근 회의 내용 알려줘")

        self.assertEqual(plan.filters["project"], "crm 프로젝트")

    def test_here_follow_up_resolves_project_from_previous_turn(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze(
            "여기서 일정 관련 내용만 다시 정리해줘",
            [{"user_query": "A 과제 최근 회의 내용 알려줘"}],
        )

        self.assertEqual(plan.intent, "recent_context_lookup")
        self.assertEqual(plan.filters["project"], "A 과제")
        self.assertIn("프로젝트: A 과제", plan.query_rewrites[-1])

    def test_follow_up_resolves_project_from_older_turn(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze(
            "일정이 공식 계획이랑 다른 부분이 있는지 확인해줘",
            [
                {
                    "user_query": "첫 질문",
                    "query_analysis": {"filters": {"project": "A 과제"}},
                },
                {"user_query": "여기서 일정 관련 내용만 다시 정리해줘"},
            ],
        )

        self.assertEqual(plan.intent, "memory_comparison")
        self.assertEqual(plan.filters["project"], "A 과제")


class PrefetchTests(unittest.TestCase):
    def test_prefetch_respects_total_top_k(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A 과제 공식 일정과 최근 결정 비교")
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)
        self.assertEqual(sum(result.allocations.values()), 8)
        self.assertLessEqual(len(result.cards), 8)
        self.assertTrue(result.cards)

    def test_prefetch_matches_normalized_project_name(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A과제 최근 회의 내용 알려줘")
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)

        self.assertTrue(result.cards)
        self.assertTrue(
            all(card.get("project") == "A 과제" for card in result.cards)
        )

    def test_prefetch_follow_up_does_not_mix_other_project(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze(
            "여기서 일정 관련 내용만 다시 정리해줘",
            [{"user_query": "A 과제 최근 회의 내용 알려줘"}],
        )
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)

        self.assertTrue(result.cards)
        self.assertNotIn("B 과제", {card.get("project") for card in result.cards})

    def test_prefetch_works_with_non_seed_project_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_root = Path(temp_dir)
            for tier in ("stm", "mtm", "ltm"):
                (memory_root / tier).mkdir()
            (memory_root / "stm" / "crm-meeting.md").write_text(
                "\n".join(
                    [
                        "---",
                        "project: CRM 프로젝트",
                        "source_type: meeting",
                        "title: CRM 프로젝트 회의",
                        "date: 2026-08-20",
                        "---",
                        "CRM 프로젝트 회의에서 고객 데이터 이관 일정을 정했다.",
                    ]
                ),
                encoding="utf-8",
            )

            store = MemoryStore(memory_root)
            plan = RuleBasedQueryAnalyzer().analyze("CRM프로젝트 최근 회의 내용 알려줘")
            result = MemoryPrefetcher(store, total_top_k=4).prefetch(plan)

            self.assertEqual(plan.filters["project"], "CRM 프로젝트")
            self.assertEqual(result.cards[0]["project"], "CRM 프로젝트")


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

    def test_session_turn_records_reasoning_and_tool_calls(self) -> None:
        base = AppConfig.load()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                base,
                project_root=Path(temp_dir),
                memory_root=PROJECT_ROOT / "memory_seed",
            )
            runtime = AgentRuntime(config, MockLLMClient(), MemoryStore(config.memory_root))
            result = runtime.run("A 과제 최근 회의 내용 알려줘")
            session_data = json.loads(
                Path(result["session_log_path"]).read_text(encoding="utf-8")
            )
            turn = session_data["turns"][-1]

            self.assertEqual(turn["query_analysis"]["filters"]["project"], "A 과제")
            self.assertGreater(turn["prefetch"]["result_count"], 0)
            self.assertGreaterEqual(len(turn["reasoning_steps"]), 1)
            self.assertGreaterEqual(len(turn["tool_calls"]), 1)

    def test_mock_runtime_uses_query_plan_filter_for_any_project(self) -> None:
        base = AppConfig.load()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            memory_root = root / "memory_seed"
            for tier in ("stm", "mtm", "ltm"):
                (memory_root / tier).mkdir(parents=True)
            (memory_root / "mtm" / "crm-project-meeting.md").write_text(
                "\n".join(
                    [
                        "---",
                        "project: CRM 프로젝트",
                        "source_type: meeting",
                        "title: CRM 프로젝트 회의",
                        "date: 2026-08-20",
                        "---",
                        "CRM 프로젝트 회의에서 데이터 이관 담당자를 정했다.",
                    ]
                ),
                encoding="utf-8",
            )
            config = replace(base, project_root=root, memory_root=memory_root)
            runtime = AgentRuntime(config, MockLLMClient(), MemoryStore(config.memory_root))
            result = runtime.run("CRM프로젝트 최근 회의 내용 알려줘")

            self.assertEqual(
                result["trace"]["tool_calls"][0]["query"],
                "CRM프로젝트 최근 회의 내용 알려줘",
            )
            session_data = json.loads(
                Path(result["session_log_path"]).read_text(encoding="utf-8")
            )
            turn = session_data["turns"][-1]
            tool_call = turn["reasoning_steps"][0]["tool_calls"][0]
            self.assertEqual(tool_call["filters"]["project"], "CRM 프로젝트")


if __name__ == "__main__":
    unittest.main()
