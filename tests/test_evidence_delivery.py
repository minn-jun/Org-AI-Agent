"""검색한 근거가 모델까지 전달되는 경로와 그 기록을 검사한다.

2026-09-28 코드 검토에서 나온 다섯 가지를 각각 막는다.

1. 컨텍스트 글자 수 제한에 걸려 빠진 카드가 기록에 남는다.
2. LTM 카드가 같은 문서의 다른 대목까지 실어 보낸다.
3. expand_evidence가 카드 발췌가 아니라 원문을 돌려준다.
4. 도구 호출 상한이 실제로 실행 횟수를 막는다.
5. 공통 문서가 과제 필터에서 감점되지 않는다.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from org_agent_mvp.agent_runtime import AgentRuntime
from org_agent_mvp.config import AppConfig
from org_agent_mvp.context_builder import ContextBuilder
from org_agent_mvp.ltm_corpus import LtmCorpus
from org_agent_mvp.memory_store import MemoryStore
from org_agent_mvp.mock_llm import MockLLMClient
from org_agent_mvp.prefetch import MemoryPrefetcher
from org_agent_mvp.query_analyzer import LLMQueryAnalyzer, RuleBasedQueryAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "memory"
QUESTION = "A 과제 예산 산정 기준이 뭐야?"


@contextmanager
def env(**values: str):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def chunk(source_id: str, title: str, text: str, index: int = 1, **meta) -> dict:
    return {
        "chunk_id": f"{source_id}-{index:04d}",
        "source_id": source_id,
        "chunk_index": index,
        "chunk_count": 1,
        "metadata": {"title": title, "source_path": f"x/{title}.pdf", **meta},
        "text": text,
    }


class CorpusCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "chunks.jsonl"
        pin = env(RETRIEVER_TOKENIZER="whitespace")
        pin.__enter__()
        self.addCleanup(pin.__exit__, None, None, None)

    def load(self, rows: list[dict], **kwargs) -> LtmCorpus:
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return LtmCorpus(self.path, **kwargs)


# --------------------------------------------------------- 1. 주입 기록 분리


class InjectionRecordTests(unittest.TestCase):
    def _cards(self) -> tuple:
        store = MemoryStore(MEMORY_FIXTURE)
        plan = RuleBasedQueryAnalyzer().analyze(QUESTION)
        cards = MemoryPrefetcher(store, total_top_k=8).prefetch(plan).cards
        self.assertGreater(len(cards), 1)
        return plan, cards

    def test_build_context_reports_what_it_injected(self) -> None:
        plan, cards = self._cards()
        built = ContextBuilder(context_mode="full").build_context(plan, cards, [])
        self.assertEqual(len(built.injected), len(cards))
        self.assertEqual(built.dropped_evidence_ids, [])
        self.assertGreater(built.evidence_chars, 0)
        self.assertEqual(len(built.injected_source_ids), len(built.injected))

    def test_dropped_cards_are_recorded_not_silent(self) -> None:
        """글자 수 제한에 걸려 빠진 카드가 기록에 남아야 한다.

        예전에는 `build()`가 문자열만 돌려줘서 런타임이 알 수 없었다.
        그래서 근거 개수에 후보 전체를 적었고, Allganize 20문항 중 15건에서
        기록과 실제가 어긋났다.
        """
        plan, cards = self._cards()
        # 첫 카드 하나만 들어갈 만큼 좁힌다.
        builder = ContextBuilder(context_mode="full", max_evidence_chars=1)
        built = builder.build_context(plan, cards, [])
        self.assertEqual(built.injected, [])
        self.assertEqual(len(built.dropped_evidence_ids), len(cards))
        # 텍스트 자체는 정상적으로 만들어진다 — 근거만 비어 있다.
        self.assertIn("[RUNTIME_CONTEXT]", built.text)

    def test_build_still_returns_text(self) -> None:
        plan, cards = self._cards()
        text = ContextBuilder(context_mode="full").build(plan, cards, [])
        self.assertIsInstance(text, str)


# ------------------------------------------------- 2. 문서당 여러 대목 전달


class MultiChunkCardTests(CorpusCase):
    ROWS = [
        chunk("doc", "보고서", "예산 총액은 표에 있다", index=1, page_nos=[1]),
        chunk("doc", "보고서", "예산 예산 산정 기준은 여기다", index=2, page_nos=[2]),
        chunk("doc", "보고서", "예산 집행 실적은 별지", index=3, page_nos=[3]),
    ]

    def test_default_keeps_one_excerpt_per_document(self) -> None:
        corpus = self.load(self.ROWS)
        (_score, card), = corpus.search(["예산"], top_k=5)
        self.assertEqual(card["additional_excerpts"], [])
        self.assertEqual(card["source_ref"]["extra_chunk_indexes"], [])

    def test_multiple_chunks_per_doc_are_carried_on_the_card(self) -> None:
        corpus = self.load(self.ROWS, chunks_per_doc=3)
        (_score, card), = corpus.search(["예산"], top_k=5)
        # 대표 청크는 그대로 최고점 청크다. 순위가 바뀌면 안 된다.
        self.assertEqual(card["source_ref"]["chunk_index"], 1)
        # 대표는 "예산"이 두 번 나온 청크다. 나머지 두 대목이 함께 실린다.
        self.assertIn("산정 기준", card["content_excerpt"])
        bodies = [item["content"] for item in card["additional_excerpts"]]
        self.assertEqual(len(bodies), 2)
        self.assertTrue(any("총액" in body for body in bodies))
        self.assertTrue(any("집행 실적" in body for body in bodies))

    def test_excerpt_length_is_configurable(self) -> None:
        long_text = "예산 " + ("가" * 2000)
        corpus = self.load([chunk("doc", "보고서", long_text)], excerpt_chars=1500)
        (_score, card), = corpus.search(["예산"], top_k=5)
        self.assertEqual(len(card["content_excerpt"]), 1500)


# --------------------------------------------------------- 3. 원문 조회


class ChunkContextTests(CorpusCase):
    def test_chunk_context_returns_whole_chunk_and_neighbors(self) -> None:
        body = "예산 " + ("나" * 1500)
        corpus = self.load([
            chunk("doc", "보고서", "앞 대목", index=1, page_nos=[1]),
            chunk("doc", "보고서", body, index=2, page_nos=[2]),
            chunk("doc", "보고서", "뒤 대목", index=3, page_nos=[3]),
        ])
        (_score, card), = corpus.search(["예산"], top_k=5)
        ref = card["source_ref"]
        # 카드 발췌는 짧다. 원문 조회는 잘리지 않는다.
        self.assertEqual(len(card["content_excerpt"]), 600)
        found = corpus.chunk_context(ref["document_id"], ref["chunk_index"])
        self.assertTrue(found["found"])
        self.assertEqual([c["chunk_index"] for c in found["chunks"]], [0, 1, 2])
        requested = [c for c in found["chunks"] if c["is_requested"]]
        self.assertEqual(len(requested[0]["content"]), len(body))
        self.assertGreater(found["chars"], len(card["content_excerpt"]))

    def test_unknown_document_is_reported_not_raised(self) -> None:
        corpus = self.load([chunk("doc", "보고서", "예산 내용")])
        self.assertFalse(corpus.chunk_context("없는문서", 0)["found"])

    def test_memory_store_falls_back_to_seed_documents(self) -> None:
        store = MemoryStore(MEMORY_FIXTURE)
        name = store.documents[0].path.name
        found = store.chunk_context(name, 0)
        self.assertTrue(found["found"])
        self.assertTrue(found["chunks"][0]["content"])


class ExpandReturnsSourceTests(unittest.TestCase):
    def test_expand_evidence_returns_more_than_the_card_excerpt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                AppConfig.load(),
                project_root=Path(temp_dir),
                memory_root=MEMORY_FIXTURE,
                context_mode="summary",
            )
            runtime = AgentRuntime(
                config=config,
                client=MockLLMClient(),
                memory_store=MemoryStore(config.memory_root),
                query_analyzer=RuleBasedQueryAnalyzer(),
            )
            plan = RuleBasedQueryAnalyzer().analyze(QUESTION)
            cards = runtime.prefetcher.prefetch(plan).cards
            index = {str(card["evidence_id"]): card for card in cards}
            known = next(iter(index))

            from org_agent_mvp.agent_runtime import AgentTrace

            message = runtime._execute_expand_evidence(
                {
                    "id": "call_1",
                    "function": {
                        "name": "expand_evidence",
                        "arguments": json.dumps(
                            {"evidence_ids": [known], "reason": "원문 확인"},
                            ensure_ascii=False,
                        ),
                    },
                },
                index,
                AgentTrace(),
            )
            payload = json.loads(message["content"])
            item = payload["expanded"][0]
            self.assertTrue(item["from_source"])
            self.assertTrue(item["content"])
            self.assertEqual(item["content_chars"], len(item["content"]))


# ------------------------------------------------------- 4. 도구 호출 상한


class ToolLimitTests(unittest.TestCase):
    def _runtime(self, temp_dir: str, max_tool_calls: int) -> AgentRuntime:
        config = replace(
            AppConfig.load(),
            project_root=Path(temp_dir),
            memory_root=MEMORY_FIXTURE,
            context_mode="summary",
            max_tool_calls=max_tool_calls,
        )
        return AgentRuntime(
            config=config,
            client=MockLLMClient(),
            memory_store=MemoryStore(config.memory_root),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )

    def test_executed_tool_calls_never_exceed_the_limit(self) -> None:
        """상한은 안내 문구가 아니라 실제 제한이어야 한다.

        예전에는 상한에 도달해도 다음 호출에 도구를 계속 넘겼고, 한 응답의
        여러 도구 요청을 전부 실행했다. 상한 3회에 8회가 실행됐다.
        """
        for limit in (1, 2, 3):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as temp_dir:
                result = self._runtime(temp_dir, limit).run(QUESTION)
                executed = len(result["trace"]["tool_calls"])
                self.assertLessEqual(executed, limit)
                self.assertTrue(result["answer"])

    def test_limit_run_still_ends_with_an_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._runtime(temp_dir, 1).run(QUESTION)
            self.assertNotEqual(result["trace"]["stopped_reason"], "loop_exhausted")


# ------------------------------------------------- 5. 근거 3분리와 인용 기록


class TraceSourceSeparationTests(unittest.TestCase):
    def test_trace_separates_retrieved_injected_and_cited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                AppConfig.load(),
                project_root=Path(temp_dir),
                memory_root=MEMORY_FIXTURE,
                context_mode="full",
            )
            runtime = AgentRuntime(
                config=config,
                client=MockLLMClient(),
                memory_store=MemoryStore(config.memory_root),
                query_analyzer=RuleBasedQueryAnalyzer(),
            )
            trace = runtime.run(QUESTION)["trace"]
            for key in (
                "retrieved_sources",
                "injected_sources",
                "tool_sources",
                "dropped_evidence_ids",
                "cited_sources",
            ):
                self.assertIn(key, trace)
            # 주입된 근거는 검색된 근거의 부분집합이다.
            self.assertTrue(
                set(trace["injected_sources"]) <= set(trace["retrieved_sources"])
            )
            # final_sources는 주입 + 도구 결과다. 후보 전체가 아니다.
            self.assertEqual(
                set(trace["final_sources"]),
                set(trace["injected_sources"]) | set(trace["tool_sources"]),
            )

    def test_injected_count_is_reported_separately_from_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = replace(
                AppConfig.load(),
                project_root=Path(temp_dir),
                memory_root=MEMORY_FIXTURE,
                context_mode="full",
                max_evidence_chars=1,
            )
            runtime = AgentRuntime(
                config=config,
                client=MockLLMClient(),
                memory_store=MemoryStore(config.memory_root),
                query_analyzer=RuleBasedQueryAnalyzer(),
            )
            log_dir = Path(temp_dir) / "turns"
            result = runtime.run(QUESTION, log_dir=log_dir)
            built = [
                event
                for event in json.loads(
                    Path(result["turn_log_path"]).read_text(encoding="utf-8")
                )["debug_events"]
                if event["event"] == "context_built"
            ]
            payload = built[0]["payload"]
            self.assertGreater(payload["candidate_count"], 0)
            self.assertEqual(payload["injected_count"], 0)
            self.assertEqual(payload["dropped_count"], payload["candidate_count"])
            self.assertEqual(result["trace"]["injected_sources"], [])


# ----------------------------------------------------- 6. 공통 문서 필터


class CommonDocumentFilterTests(CorpusCase):
    def test_common_documents_are_not_penalized_by_a_project_filter(self) -> None:
        """공통 문서는 어느 과제 질문에서도 감점되지 않는다.

        예전 LTM 코드는 비교 방향이 뒤집혀서 공통 문서를 0.3배로 감점했다.
        규정·양식·지침이 바로 그 공통 문서다.
        """
        # 본문을 다르게 둔다. 같으면 버전 접기가 본문 해시로 두 문서를 한 장으로
        # 묶어서 필터 검사가 되지 않는다.
        rows = [
            chunk("common", "운영요령", "예산 산정 기준은 공통 규정", project="공통"),
            chunk("other", "타과제계획서", "예산 산정 기준은 B 과제 계획", project="B 과제"),
        ]
        corpus = self.load(rows)
        scores = {
            card["title"]: score
            for score, card in corpus.search(
                ["예산"], filters={"project": "A 과제"}, filter_penalty=0.3, top_k=5
            )
        }
        self.assertGreater(scores["운영요령"], scores["타과제계획서"])

    def test_matching_project_is_not_penalized_either(self) -> None:
        rows = [
            chunk("mine", "우리계획서", "예산 산정 기준은 A 과제 계획", project="A 과제"),
            chunk("other", "타과제계획서", "예산 산정 기준은 B 과제 계획", project="B 과제"),
        ]
        corpus = self.load(rows)
        scores = {
            card["title"]: score
            for score, card in corpus.search(
                ["예산"], filters={"project": "A 과제"}, filter_penalty=0.3, top_k=5
            )
        }
        self.assertGreater(scores["우리계획서"], scores["타과제계획서"])


# ------------------------------------------- 7. 분석기 과제 필터 오탐 차단


class AnalyzerProjectFilterTests(unittest.TestCase):
    VOCAB = {"project": ["A 과제", "B 과제"], "source_type": ["plan"]}

    class _Client:
        """LLM이 정상 응답하되 과제명을 주지 않는 경우."""

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def chat(self, messages, tools=None, **kwargs):
            return {
                "message": {"role": "assistant", "content": json.dumps(self.payload)},
                "usage": {"total_tokens": 1},
            }

    def test_fallback_gets_the_same_vocabulary(self) -> None:
        """LLM 분석기 내부 fallback도 어휘 검증을 거쳐야 한다.

        예전에는 빈 RuleBasedQueryAnalyzer를 만들어서, "이 과제" 같은 오탐이
        실제 실행 경로에서만 필터로 들어갔다.
        """
        analyzer = LLMQueryAnalyzer(
            self._Client({}), model="x", vocabulary=self.VOCAB
        )
        self.assertEqual(
            analyzer.fallback._known_projects,
            RuleBasedQueryAnalyzer(vocabulary=self.VOCAB)._known_projects,
        )

    def test_this_project_is_not_used_as_a_filter(self) -> None:
        payload = {
            "intent": "official_knowledge_lookup",
            "memory_needed": True,
            "filters": {},
            "query_rewrites": ["이 과제의 전체 연구개발기간은 언제까지야?"],
        }
        analyzer = LLMQueryAnalyzer(
            self._Client(payload), model="x", vocabulary=self.VOCAB
        )
        plan = analyzer.analyze("이 과제의 전체 연구개발기간은 언제까지야?")
        self.assertNotIn("project", plan.filters)

    def test_known_project_from_the_model_is_kept(self) -> None:
        payload = {
            "intent": "official_knowledge_lookup",
            "memory_needed": True,
            "filters": {"project": "B 과제"},
            "query_rewrites": ["B 과제 예산"],
        }
        analyzer = LLMQueryAnalyzer(
            self._Client(payload), model="x", vocabulary=self.VOCAB
        )
        self.assertEqual(analyzer.analyze("B 과제 예산")
                         .filters["project"], "B 과제")

    def test_date_range_is_no_longer_accepted(self) -> None:
        """검색기가 해석하지 않는 필터는 통과시키지 않는다."""
        self.assertNotIn("date_range", LLMQueryAnalyzer.ALLOWED_FILTER_KEYS)

    def test_new_topic_does_not_inherit_the_previous_project(self) -> None:
        turns = [{"user_query": "A 과제 회의 내용 알려줘"}]
        plan = RuleBasedQueryAnalyzer().analyze("연차 규정 기준이 뭐야?", turns)
        self.assertNotIn("project", plan.filters)

    def test_follow_up_still_inherits(self) -> None:
        turns = [{"user_query": "A 과제 회의 내용 알려줘"}]
        plan = RuleBasedQueryAnalyzer().analyze("여기서 일정만 정리해줘", turns)
        self.assertEqual(plan.filters["project"], "A 과제")


# --------------------------------------------- 7-1. 도구 결과 예산 분리


class ToolResultBudgetTests(unittest.TestCase):
    """도구 결과는 1차 컨텍스트와 다른 예산을 쓴다.

    둘은 같은 검색 경로를 타서 카드가 같은 깊이로 만들어지는데, 비용은 다르다.
    1차 컨텍스트는 카드 8장이 한 번 실리고, 도구 결과는 10~20장이 호출마다
    다시 실린다. 2026-09-29 실측에서 입력 토큰의 46%가 도구 결과 쪽이었다.
    """

    def _runtime(self, temp_dir: str, **overrides) -> AgentRuntime:
        config = replace(
            AppConfig.load(),
            project_root=Path(temp_dir),
            memory_root=MEMORY_FIXTURE,
            context_mode="full",
            **overrides,
        )
        return AgentRuntime(
            config=config,
            client=MockLLMClient(),
            memory_store=MemoryStore(config.memory_root),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )

    @staticmethod
    def _card(body: str, extras: int) -> dict:
        return {
            "evidence_id": "ev_ltm_x",
            "content_excerpt": body,
            "additional_excerpts": [
                {"chunk_index": i, "page_nos": [i], "content": body}
                for i in range(1, extras + 1)
            ],
            "source_ref": {"document_id": "x", "extra_chunk_indexes": list(range(1, extras + 1))},
        }

    def test_tool_result_is_trimmed_to_its_own_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(
                temp_dir,
                ltm_chunks_per_doc=4,
                ltm_excerpt_chars=2000,
                tool_result_chunks_per_doc=1,
                tool_result_excerpt_chars=600,
            )
            raw = {"result_count": 1, "results": [self._card("가" * 2000, 3)]}
            out = runtime._trim_tool_result(raw)
            card = out["results"][0]
            self.assertEqual(len(card["content_excerpt"]), 600)
            self.assertEqual(card["additional_excerpts"], [])
            self.assertEqual(card["source_ref"]["extra_chunk_indexes"], [])

    def test_trimming_does_not_touch_the_original_cards(self) -> None:
        """prefetch가 같은 카드 객체를 쓸 수 있으므로 원본을 고치면 안 된다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir, tool_result_excerpt_chars=100)
            original = self._card("나" * 900, 2)
            runtime._trim_tool_result({"results": [original]})
            self.assertEqual(len(original["content_excerpt"]), 900)
            self.assertEqual(len(original["additional_excerpts"]), 2)

    def test_tool_budget_can_be_raised(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(
                temp_dir, tool_result_chunks_per_doc=3, tool_result_excerpt_chars=1500
            )
            card = runtime._trim_tool_result(
                {"results": [self._card("다" * 2000, 4)]}
            )["results"][0]
            self.assertEqual(len(card["content_excerpt"]), 1500)
            self.assertEqual(len(card["additional_excerpts"]), 2)

    def test_default_budget_matches_the_old_fixed_values(self) -> None:
        """기본값은 예산 설정이 생기기 전과 같아야 한다."""
        config = AppConfig.load()
        self.assertEqual(config.tool_result_excerpt_chars, 600)
        self.assertEqual(config.tool_result_chunks_per_doc, 1)


# ------------------------------- 7-2. 도구 결과 중복 제거와 top_k 상한


class ToolResultDedupeTests(unittest.TestCase):
    """재검색이 **이미 전달한 문서**를 다시 실어 오는 것을 막는다.

    2026-09-29 실측(20문항): 도구 카드 96장 중 66장(69%)이 이미 1차 컨텍스트나
    앞선 도구 호출로 전달한 문서였다. q_180은 5장 전부가 중복이었다.
    """

    def _runtime(self, temp_dir: str, **overrides) -> AgentRuntime:
        overrides.setdefault("tool_result_dedupe", True)
        config = replace(
            AppConfig.load(),
            project_root=Path(temp_dir),
            memory_root=MEMORY_FIXTURE,
            context_mode="full",
            **overrides,
        )
        return AgentRuntime(
            config=config,
            client=MockLLMClient(),
            memory_store=MemoryStore(config.memory_root),
            query_analyzer=RuleBasedQueryAnalyzer(),
        )

    @staticmethod
    def _result(doc_ids: list[str]) -> dict:
        return {
            "result_count": len(doc_ids),
            "results": [
                {
                    "evidence_id": f"ev_ltm_{d}",
                    "content_excerpt": f"{d} 본문",
                    "source_ref": {"document_id": d},
                }
                for d in doc_ids
            ],
        }

    def _trace(self, injected: list[str], tool: list[str] | None = None):
        from org_agent_mvp.agent_runtime import AgentTrace

        trace = AgentTrace()
        trace.injected_sources = list(injected)
        trace.tool_sources = list(tool or [])
        return trace

    def test_already_injected_documents_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir)
            out = runtime._trim_tool_result(
                self._result(["a", "b", "c"]), self._trace(["a", "c"])
            )
            ids = [c["source_ref"]["document_id"] for c in out["results"]]
            self.assertEqual(ids, ["b"])
            self.assertEqual(out["duplicates_removed"], 2)
            self.assertEqual(out["result_count"], 1)

    def test_documents_from_earlier_tool_calls_count_as_delivered(self) -> None:
        """같은 턴의 앞선 재검색이 이미 가져온 문서도 중복이다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir)
            out = runtime._trim_tool_result(
                self._result(["a", "b"]), self._trace(["x"], tool=["a"])
            )
            ids = [c["source_ref"]["document_id"] for c in out["results"]]
            self.assertEqual(ids, ["b"])

    def test_duplicates_within_one_result_are_removed_too(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir)
            out = runtime._trim_tool_result(
                self._result(["a", "a", "b"]), self._trace([])
            )
            ids = [c["source_ref"]["document_id"] for c in out["results"]]
            self.assertEqual(ids, ["a", "b"])

    def test_backfill_keeps_the_requested_count(self) -> None:
        """중복을 뺀 뒤에도 요청한 장수가 남아야 한다.

        중복만 지우고 끝내면 결과가 비고, 모델은 "찾아도 아무것도 없다"로 읽어
        한 번 더 검색한다. 그래서 런타임이 넉넉히 받아 오고 여기서 잘라 준다.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir)
            fetched = self._result(["a", "b", "c", "d", "e", "f"])
            out = runtime._trim_tool_result(fetched, self._trace(["a", "b"]), limit=3)
            ids = [c["source_ref"]["document_id"] for c in out["results"]]
            self.assertEqual(ids, ["c", "d", "e"])   # 빈 목록이 아니다

    def test_dedupe_is_off_by_default(self) -> None:
        config = AppConfig.load()
        self.assertFalse(config.tool_result_dedupe)
        self.assertEqual(config.tool_result_top_k_cap, 0)

    def test_dedupe_off_keeps_every_card(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir, tool_result_dedupe=False)
            out = runtime._trim_tool_result(
                self._result(["a", "b"]), self._trace(["a", "b"])
            )
            self.assertEqual(len(out["results"]), 2)
            self.assertEqual(out["duplicates_removed"], 0)

    def test_limit_caps_the_card_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(temp_dir, tool_result_dedupe=False)
            out = runtime._trim_tool_result(
                self._result(["a", "b", "c", "d"]), self._trace([]), limit=2
            )
            self.assertEqual(len(out["results"]), 2)

    def test_top_k_cap_limits_what_the_model_can_ask(self) -> None:
        """모델이 top_k=20을 요청해도 상한까지만 돌려준다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = self._runtime(
                temp_dir, tool_result_dedupe=False, tool_result_top_k_cap=3
            )
            from org_agent_mvp.agent_runtime import AgentTrace

            message = runtime._execute_tool_call(
                {
                    "id": "call_1",
                    "function": {
                        "name": "retrieve_memory",
                        "arguments": json.dumps(
                            {"tier": "all", "query": "예산 산정 기준", "top_k": 20},
                            ensure_ascii=False,
                        ),
                    },
                },
                set(),
                {},
                AgentTrace(),
            )
            payload = json.loads(message["content"])
            self.assertLessEqual(len(payload["results"]), 3)
            self.assertEqual(payload["requested_top_k"], 20)
            self.assertEqual(payload["effective_top_k"], 3)


# ------------------------------------------ 8. 평가 기준선 = 서비스 기본값


class BenchBaselineTests(unittest.TestCase):
    """`bench_allganize.py`의 BASE_ENV가 코드 기본값과 같아야 한다.

    이 표는 docstring에 "= 코드 기본값"이라고 적혀 있는데 실제로는
    2026-09-16의 기본값 변경을 따라오지 않았다. README의 기본 평가 명령이
    서비스와 다른 검색기를 재고 있었다. 값이 갈리면 여기서 걸린다.
    """

    def _base_env(self) -> dict[str, str]:
        import importlib.util

        path = PROJECT_ROOT / "scripts" / "bench_allganize.py"
        spec = importlib.util.spec_from_file_location("bench_allganize", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.BASE_ENV

    def test_base_env_matches_code_defaults(self) -> None:
        from org_agent_mvp import ltm_corpus, memory_store, scoring, tokenizer

        base = self._base_env()
        with env(**base):
            self.assertEqual(tokenizer.tokenizer_name(), base["RETRIEVER_TOKENIZER"])
            self.assertEqual(scoring.scorer_name(), base["RETRIEVER_SCORER"])
            self.assertEqual(scoring.bm25_k1(), float(base["RETRIEVER_BM25_K1"]))
            self.assertEqual(ltm_corpus.title_bonus_mode(), base["RETRIEVER_TITLE_BONUS"])
            with_base = memory_store.query_expansions()
        # 기본값(환경변수 없음)과 같은 결과가 나와야 한다.
        for key in base:
            os.environ.pop(key, None)
        self.assertEqual(tokenizer.tokenizer_name(), base["RETRIEVER_TOKENIZER"])
        self.assertEqual(scoring.scorer_name(), base["RETRIEVER_SCORER"])
        self.assertEqual(scoring.bm25_k1(), float(base["RETRIEVER_BM25_K1"]))
        self.assertEqual(ltm_corpus.title_bonus_mode(), base["RETRIEVER_TITLE_BONUS"])
        self.assertEqual(with_base, memory_store.query_expansions())


if __name__ == "__main__":
    unittest.main()


# ------------------------------------- 7. 시드 문서 발췌를 따로 정한다 (2026-10-07)


class SeedExcerptTests(unittest.TestCase):
    """시드 문서와 청크 코퍼스의 발췌 길이를 나눠서 정할 수 있어야 한다.

    둘은 모양이 다르다. 청크 카드는 `LTM_CHUNKS_PER_DOC`으로 대목을 늘릴 수
    있지만, 시드 문서는 **파일 하나가 카드 하나**라 그 손잡이가 걸리지 않는다.
    한 값을 공유하면 진행 과제 시드를 손보려고 Allganize 설정까지 움직이게 된다.
    """

    def _store(self, **overrides):
        from org_agent_mvp.memory_store import build_memory_store

        config = replace(
            AppConfig.load(), memory_root=MEMORY_FIXTURE, ltm_corpus_path=None, **overrides
        )
        return build_memory_store(config)

    def test_default_follows_the_chunk_setting(self) -> None:
        self.assertEqual(AppConfig.load().seed_excerpt_chars, 0)
        store = self._store(ltm_excerpt_chars=777)
        self.assertEqual(store.excerpt_chars, 777)

    def test_seed_value_overrides_it(self) -> None:
        store = self._store(ltm_excerpt_chars=600, seed_excerpt_chars=5000)
        self.assertEqual(store.excerpt_chars, 5000)

    def test_longer_excerpt_delivers_more_of_the_document(self) -> None:
        """긴 문서에서 발췌 길이가 실제로 전달량을 바꾼다.

        fixture 문서는 200자 안쪽이라 잘릴 것이 없다. 진행 과제 시드의 LTM
        문서는 평균 21,582자이므로, 그 길이를 흉내 낸 문서를 따로 만든다.
        """
        from org_agent_mvp.memory_store import build_memory_store

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "seed"
            (root / "ltm").mkdir(parents=True)
            chunk = "마일스톤 점검 결과와 예산 집행 내역을 적는다. "
            body = "\n\n".join(
                f"## 항목 {index}\n{chunk * 3}" for index in range(60)
            )
            self.assertGreater(len(body), 4000)
            (root / "ltm" / "plan.md").write_text(
                "---\nmemory_tier: ltm\nproject: A 과제\n"
                "title: A 과제 계획서\ndate: 2026-06-03\n---\n\n" + body,
                encoding="utf-8",
            )

            def excerpt(seed_chars: int) -> str:
                config = replace(
                    AppConfig.load(),
                    memory_root=root,
                    ltm_corpus_path=None,
                    ltm_excerpt_chars=600,
                    seed_excerpt_chars=seed_chars,
                )
                result = build_memory_store(config).retrieve(
                    tier="ltm", query="마일스톤 예산 집행", filters={}, top_k=1
                )
                return str(result["results"][0]["content_excerpt"])

            short, long = excerpt(200), excerpt(4000)
            self.assertEqual(len(short), 200)
            self.assertGreater(len(long), len(short))
