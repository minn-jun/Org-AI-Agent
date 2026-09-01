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
from org_agent_mvp.query_analyzer import LLMQueryAnalyzer, RuleBasedQueryAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeAnalyzerClient:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        if self.error:
            raise self.error
        return {
            "message": {"role": "assistant", "content": self.content},
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
                "estimated": False,
            },
        }


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

    def test_session_summary_uses_session_only_plan(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze(
            "아까 말한거 요약해줘",
            [{"user_query": "A 과제 공식 일정 알려줘", "answer_summary": "공식 제출일 확인"}],
        )

        self.assertEqual(plan.intent, "session_context_answer")
        self.assertEqual(plan.answer_source, "session_only")
        self.assertTrue(plan.use_session_context)
        self.assertFalse(plan.memory_needed)
        self.assertEqual(plan.memory_weights, {"stm": 0.0, "mtm": 0.0, "ltm": 0.0})

    def test_session_referenced_fact_still_uses_memory(self) -> None:
        plan = RuleBasedQueryAnalyzer().analyze(
            "아까 말한 일정의 담당자는?",
            [{"user_query": "A 과제 공식 일정 알려줘", "answer_summary": "공식 제출일 확인"}],
        )

        self.assertEqual(plan.intent, "recent_context_lookup")
        self.assertEqual(plan.answer_source, "memory_prefetch")
        self.assertTrue(plan.use_session_context)
        self.assertTrue(plan.memory_needed)

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

    def test_llm_query_analyzer_uses_model_json_plan(self) -> None:
        client = FakeAnalyzerClient(
            json.dumps(
                {
                    "intent": "recent_context_lookup",
                    "can_answer_directly": False,
                    "memory_needed": True,
                    "answer_source": "memory_prefetch",
                    "use_session_context": False,
                    "memory_weights": {"stm": 0.7, "mtm": 0.2, "ltm": 0.1},
                    "query_rewrites": ["A과제 최근 회의 내용 알려줘"],
                    "filters": {"project": "A과제"},
                    "reason": "최근 회의 맥락 확인 필요",
                },
                ensure_ascii=False,
            )
        )

        plan = LLMQueryAnalyzer(client, model="small-model").analyze(
            "A과제 최근 회의 내용 알려줘"
        )

        self.assertEqual(plan.intent, "recent_context_lookup")
        self.assertEqual(plan.filters["project"], "A 과제")
        self.assertAlmostEqual(sum(plan.memory_weights.values()), 1.0)
        self.assertEqual(client.calls[0]["model"], "small-model")

    def test_llm_query_analyzer_keeps_search_when_model_skips_it(self) -> None:
        """규칙은 검색이 필요하다는데 LLM만 session_only면 검색을 유지한다."""
        client = FakeAnalyzerClient(
            json.dumps(
                {
                    "intent": "session_context_answer",
                    "can_answer_directly": True,
                    "memory_needed": False,
                    "answer_source": "session_only",
                    "use_session_context": True,
                    "memory_weights": {"stm": 0.0, "mtm": 0.0, "ltm": 0.0},
                    "query_rewrites": ["A 과제 예산 산정 기준이 뭐야?"],
                    "filters": {},
                    "reason": "세션만으로 답변 가능",
                },
                ensure_ascii=False,
            )
        )

        plan = LLMQueryAnalyzer(client, model="small-model").analyze(
            "A 과제 예산 산정 기준이 뭐야?"
        )

        self.assertTrue(plan.memory_needed)
        self.assertEqual(plan.answer_source, "memory_prefetch")
        self.assertFalse(plan.can_answer_directly)
        self.assertGreater(sum(plan.memory_weights.values()), 0)

    def test_llm_query_analyzer_rejects_unknown_and_translated_filters(self) -> None:
        """스키마 밖 필터는 버리고, 과제명은 규칙 기반 추출값을 지킨다."""
        client = FakeAnalyzerClient(
            json.dumps(
                {
                    "intent": "official_knowledge_lookup",
                    "can_answer_directly": False,
                    "memory_needed": True,
                    "answer_source": "memory_prefetch",
                    "use_session_context": False,
                    "memory_weights": {"stm": 0.1, "mtm": 0.2, "ltm": 0.7},
                    "query_rewrites": ["A 과제 공식 제출일"],
                    "filters": {"project": "assignment_a", "type": "deadline"},
                    "reason": "공식 기준 확인 필요",
                },
                ensure_ascii=False,
            )
        )

        plan = LLMQueryAnalyzer(client, model="small-model").analyze(
            "A 과제 공식 제출일 알려줘"
        )

        self.assertEqual(plan.filters["project"], "A 과제")
        self.assertNotIn("type", plan.filters)

    def test_llm_query_analyzer_falls_back_on_error(self) -> None:
        client = FakeAnalyzerClient(error=RuntimeError("rate limited"))

        plan = LLMQueryAnalyzer(client, model="small-model").analyze(
            "A 과제의 최근 일정이 공식 계획과 충돌해?"
        )

        self.assertEqual(plan.intent, "memory_comparison")
        self.assertIn("fallback", plan.reason)


class PrefetchTests(unittest.TestCase):
    def test_prefetch_respects_total_top_k(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A 과제 공식 일정과 최근 결정 비교")
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)
        self.assertLessEqual(len(result.cards), 8)
        self.assertTrue(result.cards)

    def test_prefetch_uses_tier_prior_instead_of_quota(self) -> None:
        """tier 가중치는 자리 수가 아니라 점수 배수로만 작용해야 한다."""
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A 과제 공식 일정과 최근 결정 비교")
        result = MemoryPrefetcher(store, total_top_k=8, alpha=1.0).prefetch(plan)

        self.assertEqual(result.tier_priors, plan.memory_weights)
        for card in result.cards:
            expected = round(
                card["normalized_score"] * (1.0 + card["tier_prior"]), 4
            )
            self.assertAlmostEqual(card["final_score"], expected, places=3)
        scores = [card["final_score"] for card in result.cards]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_prefetch_cuts_by_relative_threshold(self) -> None:
        """최고점 대비 비율 아래는 버리고, 최소 건수는 보장한다."""
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A 과제 공식 일정과 최근 결정 비교")
        result = MemoryPrefetcher(
            store, total_top_k=8, cut_ratio=0.9, min_cards=2
        ).prefetch(plan)

        self.assertGreaterEqual(len(result.cards), 2)
        top = result.cards[0]["final_score"]
        self.assertAlmostEqual(result.cut_threshold, round(top * 0.9, 4), places=3)

    def test_prefetch_alpha_zero_ignores_tier(self) -> None:
        """alpha=0이면 tier를 무시하고 관련성 순수 순위가 된다 (ablation arm)."""
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A 과제 공식 일정과 최근 결정 비교")
        result = MemoryPrefetcher(store, total_top_k=8, alpha=0.0).prefetch(plan)

        for card in result.cards:
            self.assertAlmostEqual(
                card["final_score"], card["normalized_score"], places=3
            )

    def test_prefetch_skips_search_when_memory_not_needed(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze(
            "아까 말한거 요약해줘",
            [{"user_query": "A 과제 최근 회의 내용 알려줘"}],
        )
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)

        self.assertFalse(plan.memory_needed)
        self.assertEqual(result.cards, [])
        self.assertEqual(result.pool_per_tier, 0)

    def test_prefetch_matches_normalized_project_name(self) -> None:
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        plan = RuleBasedQueryAnalyzer().analyze("A과제 최근 회의 내용 알려줘")
        result = MemoryPrefetcher(store, total_top_k=8).prefetch(plan)

        self.assertTrue(result.cards)
        # 공통 기준 문서는 어느 과제 질문에서든 후보가 된다.
        # 막아야 하는 것은 "다른 과제"가 섞이는 경우다.
        projects = {card.get("project") for card in result.cards}
        self.assertIn("A 과제", projects)
        self.assertTrue(projects <= {"A 과제", "공통"}, f"다른 과제가 섞였다: {projects}")

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

    def test_mock_runtime_answers_session_summary_without_prefetch(self) -> None:
        base = AppConfig.load()
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                base,
                project_root=Path(temp_dir),
                memory_root=PROJECT_ROOT / "memory_seed",
            )
            runtime = AgentRuntime(config, MockLLMClient(), MemoryStore(config.memory_root))
            first = runtime.run("A 과제 최근 일정이 공식 계획과 충돌해?")
            second = runtime.run("아까 말한거 요약해줘", session_id=first["session_id"])

            self.assertEqual(second["trace"]["query_analysis"]["intent"], "session_context_answer")
            self.assertFalse(second["trace"]["query_analysis"]["memory_needed"])
            self.assertEqual(second["trace"]["prefetch"]["result_count"], 0)
            self.assertEqual(second["trace"]["tool_calls"], [])
            self.assertIn("앞선 대화 요약", second["answer"])

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
