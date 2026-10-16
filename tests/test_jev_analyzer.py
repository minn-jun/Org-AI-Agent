"""Jev 질의 분석기 — 7분기 판정과 저장소 선택 (2026-10-16).

여기서 고정하는 것은 두 가지다.

1. **고른 분기의 계층만 1.0, 나머지는 0.0.** 소수 가중치를 섞지 않는다.
   기존 분석기처럼 `stm 0.10 · mtm 0.20 · ltm 0.70`이 나오면 안 된다.
2. **가중치 0인 계층은 검색조차 하지 않는다.** 0은 "조금 덜 본다"가 아니라
   "보지 않는다"는 뜻이다.

API는 부르지 않는다. 가짜 클라이언트로 응답 모양만 흉내 낸다.
실제 호출 확인은 scripts/bench_jev_routing.py가 한다.
"""

from __future__ import annotations

import os
import unittest

from org_mem.jev_analyzer import BRANCH_TIERS, QUESTIONS, JevQueryAnalyzer
from org_mem.prefetch import MemoryPrefetcher
from org_mem.query_analyzer import RuleBasedQueryAnalyzer


def jev_response(
    branch: str = "ltm",
    *,
    source: str = "memory_prefetch",
    comparison: float = 0.0,
    confidence: float = 0.8,
) -> dict:
    """실제 응답과 같은 모양. 2026-10-16 실측 응답을 본으로 삼았다."""
    return {
        "model": "typesafe/jev-1.14-20261014",
        "latency_s": 0.52,
        "answers": {
            "tier_branch": {
                "type": "choice",
                "choice": branch,
                "probabilities": {name: 0.0 for name in BRANCH_TIERS} | {branch: confidence},
                "confidence": confidence,
            },
            "answer_source": {
                "type": "choice",
                "choice": source,
                "probabilities": {source: 0.9},
                "confidence": 0.9,
            },
            "is_comparison": {"type": "noul", "noul": comparison},
        },
        "usage": {
            "prompt_tokens": 480,
            "completion_tokens": 60,
            "total_tokens": 540,
            "estimated": False,
            "cost_usd": 2.016e-05,
        },
    }


class FakeJevClient:
    """`decide()`만 흉내 내는 가짜. 보낸 질문을 기록한다."""

    model = "typesafe/jev-1.14"

    def __init__(self, response: dict | None = None, error: Exception | None = None):
        self._response = response or jev_response()
        self._error = error
        self.calls: list[tuple] = []

    def decide(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))
        if self._error is not None:
            raise self._error
        return self._response


# ------------------------------------------- 1. 분기 -> 저장소 선택


class BranchToTierTests(unittest.TestCase):
    def test_every_branch_selects_exactly_its_tiers(self) -> None:
        for branch, expected in BRANCH_TIERS.items():
            with self.subTest(branch=branch):
                analyzer = JevQueryAnalyzer(FakeJevClient(jev_response(branch)))
                plan = analyzer.analyze("과제 예산이 얼마야?")
                chosen = tuple(
                    tier for tier in ("stm", "mtm", "ltm")
                    if plan.memory_weights[tier] > 0
                )
                self.assertEqual(chosen, expected)

    def test_weights_are_on_or_off_never_fractional(self) -> None:
        """소수 가중치가 섞이면 저장소 선택이 아니라 순위 보정으로 되돌아간다."""
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("mtm_ltm")))
        plan = analyzer.analyze("최근 회의 자료랑 공식 계획서 비교해줘")
        self.assertEqual(
            plan.memory_weights, {"stm": 0.0, "mtm": 1.0, "ltm": 1.0}
        )

    def test_selected_tiers_share_one_weight(self) -> None:
        """고른 계층이 모두 같은 값이라 (1 + alpha × prior) 곱이 순위를 바꾸지 않는다."""
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("all")))
        plan = analyzer.analyze("아무거나")
        self.assertEqual(set(plan.memory_weights.values()), {1.0})

    def test_unknown_branch_falls_back_to_all(self) -> None:
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("stm_mtm_ltm_extra")))
        plan = analyzer.analyze("과제 일정 알려줘")
        self.assertTrue(all(value == 1.0 for value in plan.memory_weights.values()))
        self.assertTrue(analyzer.last_decision["overrides"])


# --------------------------------- 2. 선택한 계층만 실제로 검색되는지


class SearchOnlySelectedTiersTests(unittest.TestCase):
    class _Store:
        """어느 계층이 검색됐는지 기록하는 가짜 저장소."""

        def __init__(self) -> None:
            self.asked: list[str] = []

        def retrieve(self, *, tier, query, filters, top_k):
            self.asked.append(tier)
            return {
                "results": [
                    {
                        "evidence_id": f"ev_{tier}",
                        "tier": tier,
                        "title": f"{tier} 문서",
                        "retrieval_score": 5.0,
                        "source_ref": {"document_id": f"doc_{tier}"},
                        "excerpt": "본문",
                    }
                ],
                "result_count": 1,
            }

    def _asked_tiers(self, branch: str) -> list[str]:
        store = self._Store()
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response(branch)))
        plan = analyzer.analyze("질문")
        MemoryPrefetcher(store, total_top_k=8).prefetch(plan)
        return store.asked

    def test_single_tier_branch_searches_one_store(self) -> None:
        self.assertEqual(self._asked_tiers("ltm"), ["ltm"])

    def test_pair_branch_searches_two_stores(self) -> None:
        self.assertEqual(self._asked_tiers("stm_ltm"), ["stm", "ltm"])

    def test_all_branch_searches_every_store(self) -> None:
        self.assertEqual(self._asked_tiers("all"), ["stm", "mtm", "ltm"])

    def test_rule_analyzer_still_searches_every_store(self) -> None:
        """규칙 기반은 0을 내보내지 않으므로 기존 동작이 그대로여야 한다."""
        store = self._Store()
        plan = RuleBasedQueryAnalyzer().analyze("공식 계획서 최종 승인 기준이 뭐야?")
        MemoryPrefetcher(store, total_top_k=8).prefetch(plan)
        self.assertEqual(store.asked, ["stm", "mtm", "ltm"])


# ----------------------------------------- 3. 검색 생략과 규칙 방어


class SkipSearchDefenseTests(unittest.TestCase):
    def test_direct_answer_skips_search_when_rule_agrees(self) -> None:
        # "개념"은 규칙의 DIRECT_MARKERS라 규칙도 검색이 필요 없다고 본다.
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("ltm", source="direct_answer")))
        plan = analyzer.analyze("트랜스포머 개념이 뭐야?")
        self.assertFalse(plan.memory_needed)
        self.assertEqual(set(plan.memory_weights.values()), {0.0})

    def test_rule_restores_search_when_jev_wants_to_skip(self) -> None:
        """규칙이 검색이 필요하다고 보면 Jev의 생략 판단을 덮어쓴다.

        필요한 검색을 건너뛰면 근거 없이 답하게 되어 실패 방향이 훨씬 나쁘다.
        """
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("mtm", source="direct_answer")))
        plan = analyzer.analyze("지난주 회의록에서 결정된 담당자가 누구야?")
        self.assertTrue(plan.memory_needed)
        self.assertEqual(plan.answer_source, "memory_prefetch")
        self.assertEqual(plan.memory_weights, {"stm": 0.0, "mtm": 1.0, "ltm": 0.0})

    def test_override_is_recorded_separately_from_jev_answer(self) -> None:
        """덮어쓴 사실을 남기지 않으면 Jev가 무엇을 골랐는지 되읽을 수 없다."""
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("mtm", source="session_only")))
        analyzer.analyze("지난주 회의록에서 결정된 담당자가 누구야?")
        decision = analyzer.last_decision
        self.assertEqual(decision["jev"]["answer_source"], "session_only")
        self.assertEqual(decision["applied"]["answer_source"], "memory_prefetch")
        self.assertTrue(decision["overrides"])


# -------------------------------------------------- 4. 실패 시 복귀


class FallbackTests(unittest.TestCase):
    def test_api_failure_returns_rule_plan(self) -> None:
        client = FakeJevClient(error=RuntimeError("Jev HTTP 503"))
        analyzer = JevQueryAnalyzer(client)
        plan = analyzer.analyze("공식 계획서 기준이 뭐야?")
        self.assertTrue(analyzer.last_fallback_used)
        self.assertTrue(plan.memory_needed)
        # 규칙 가중치로 돌아갔으므로 세 계층이 모두 0보다 크다.
        self.assertTrue(all(value > 0 for value in plan.memory_weights.values()))
        self.assertIn("Jev 호출 실패", plan.reason)

    def test_failure_does_not_stop_search(self) -> None:
        """판정 모델이 죽었을 때 검색이 멈추는 것이 가장 나쁜 실패다."""
        store = SearchOnlySelectedTiersTests._Store()
        analyzer = JevQueryAnalyzer(FakeJevClient(error=RuntimeError("timeout")))
        plan = analyzer.analyze("과제 예산 산정 근거가 뭐야?")
        MemoryPrefetcher(store, total_top_k=8).prefetch(plan)
        self.assertEqual(store.asked, ["stm", "mtm", "ltm"])


# -------------------------------------------------- 5. intent 파생


class IntentTests(unittest.TestCase):
    def test_single_tier_branches_map_to_their_intent(self) -> None:
        expected = {
            "stm": "recent_context_lookup",
            "mtm": "working_document_lookup",
            "ltm": "official_knowledge_lookup",
        }
        for branch, intent in expected.items():
            with self.subTest(branch=branch):
                analyzer = JevQueryAnalyzer(FakeJevClient(jev_response(branch)))
                self.assertEqual(analyzer.analyze("질문").intent, intent)

    def test_comparison_noul_wins_over_branch_intent(self) -> None:
        analyzer = JevQueryAnalyzer(
            FakeJevClient(jev_response("stm_ltm", comparison=0.91))
        )
        self.assertEqual(analyzer.analyze("질문").intent, "memory_comparison")

    def test_pair_branch_without_comparison_is_generic_lookup(self) -> None:
        analyzer = JevQueryAnalyzer(FakeJevClient(jev_response("mtm_ltm")))
        self.assertEqual(analyzer.analyze("질문").intent, "organization_memory_lookup")


# ------------------------------------------- 6. 보내는 입력 모양


class StateTests(unittest.TestCase):
    def test_state_carries_question_and_recent_turns_only(self) -> None:
        """LLM 분석기와 같은 정보만 넣는다. 이전 답변 본문은 넣지 않는다.

        한쪽에만 더 주면 모델 비교가 아니라 입력량 비교가 된다.
        """
        client = FakeJevClient()
        analyzer = JevQueryAnalyzer(client)
        turns = [
            {
                "user_query": "이번 주 세미나 언제야?",
                "query_intent": "recent_context_lookup",
                "answer_summary": "8월 27일로 정해졌습니다",
                "source_ids": ["doc_a"],
                "query_analysis": {"filters": {"project": "진행 과제"}},
            }
        ]
        analyzer.analyze("그럼 그게 계획서랑 맞아?", turns)
        state = client.calls[0][0]
        self.assertEqual(set(state), {"사용자_질문", "최근_대화"})
        carried = state["최근_대화"][0]
        self.assertEqual(set(carried), {"질문", "의도", "과제", "출처"})
        self.assertNotIn("answer_summary", carried)

    def test_state_omits_recent_turns_on_first_turn(self) -> None:
        client = FakeJevClient()
        JevQueryAnalyzer(client).analyze("첫 질문")
        self.assertEqual(set(client.calls[0][0]), {"사용자_질문"})

    def test_questions_match_the_api_contract(self) -> None:
        """type은 noul/choice/score/box 중 하나, choice는 criteria가 필수다."""
        for name, question in QUESTIONS.items():
            with self.subTest(question=name):
                self.assertIn(question["type"], {"noul", "choice", "score", "box"})
                self.assertTrue(question["instructions"])
                if question["type"] == "choice":
                    self.assertTrue(question["criteria"])

    def test_branch_criteria_cover_every_branch(self) -> None:
        from org_mem.jev_analyzer import BRANCH_CRITERIA

        self.assertEqual(set(BRANCH_CRITERIA), set(BRANCH_TIERS))


if __name__ == "__main__":
    unittest.main()
