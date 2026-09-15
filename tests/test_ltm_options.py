"""검색기 옵션 테스트: 청크 단위 순위·페이지, 제목 가산점 방식, BM25+, k1 설정, 문서 메타데이터 끄기.

전부 기본값에서는 기존 동작과 같아야 한다. 기본값이 아닌 경로만 여기서 확인한다.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from org_agent_mvp import scoring
from org_agent_mvp.ltm_corpus import LtmCorpus, title_bonus_mode


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

    def load(self, rows: list[dict]) -> LtmCorpus:
        with self.path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return LtmCorpus(self.path)


class ChunkSearchTests(CorpusCase):
    def test_search_chunks_returns_every_matching_chunk_with_pages(self) -> None:
        corpus = self.load([
            chunk("doc", "보고서", "총사업비 첫 페이지", index=1, page_nos=[1]),
            chunk("doc", "보고서", "총사업비 총사업비 둘째와 셋째", index=2, page_nos=[2, 3]),
            chunk("other", "메모", "관련 없음", index=1, page_no=5),
        ])
        results = corpus.search_chunks(["총사업비"], top_k=10)
        self.assertEqual(len(results), 2)          # 문서로 접지 않는다
        self.assertEqual(results[0][1]["page_nos"], [2, 3])
        self.assertEqual({tuple(card["page_nos"]) for _, card in results}, {(1,), (2, 3)})

    def test_document_card_carries_page_nos_and_page_no_fallback(self) -> None:
        corpus = self.load([chunk("doc", "메모", "이월 신청", page_no=7)])
        _, card = corpus.search(["이월"], top_k=1)[0]
        self.assertEqual(card["source_ref"]["page_nos"], [7])

    def test_chunk_scores_match_document_scores(self) -> None:
        corpus = self.load([
            chunk("a", "이월 요청서", "이월 사유", index=1),
            chunk("b", "보고서", "이월 이월 금액", index=1),
        ])
        by_doc = {card["source_ref"]["document_id"]: score for score, card in corpus.search(["이월"], top_k=5)}
        by_chunk = {card["document_id"]: score for score, card in corpus.search_chunks(["이월"], top_k=5)}
        self.assertEqual(by_doc, by_chunk)


class TitleBonusTests(CorpusCase):
    rows = [
        chunk("titled", "이월 승인 요청서", "무관한 본문 내용"),
        chunk("body", "다른 문서", "이월 이월 이월 이월"),
    ]

    def test_mode_defaults_to_add_and_rejects_unknown(self) -> None:
        with env(RETRIEVER_TITLE_BONUS="weird"):
            self.assertEqual(title_bonus_mode(), "add")

    def test_none_removes_the_title_boost(self) -> None:
        corpus = self.load(self.rows)
        with env(RETRIEVER_TITLE_BONUS="add"):
            add = {c["document_id"]: s for s, c in corpus.search_chunks(["이월"], top_k=5)}
        with env(RETRIEVER_TITLE_BONUS="none"):
            none = {c["document_id"]: s for s, c in corpus.search_chunks(["이월"], top_k=5)}
        self.assertAlmostEqual(add["titled"] - none["titled"], scoring_title_bonus())
        self.assertEqual(add["body"], none["body"])

    def test_mult_is_bounded_to_ten_percent(self) -> None:
        corpus = self.load(self.rows)
        with env(RETRIEVER_TITLE_BONUS="none"):
            none = {c["document_id"]: s for s, c in corpus.search_chunks(["이월"], top_k=5)}
        with env(RETRIEVER_TITLE_BONUS="mult"):
            mult = {c["document_id"]: s for s, c in corpus.search_chunks(["이월"], top_k=5)}
        self.assertAlmostEqual(mult["titled"], none["titled"] * 1.1)
        self.assertAlmostEqual(mult["body"], none["body"])


def scoring_title_bonus() -> float:
    from org_agent_mvp.ltm_corpus import TITLE_BONUS
    return TITLE_BONUS


class Bm25PlusTests(unittest.TestCase):
    def test_scorer_name_accepts_bm25plus(self) -> None:
        with env(RETRIEVER_SCORER="bm25plus"):
            self.assertEqual(scoring.scorer_name(), "bm25plus")
            self.assertEqual(scoring.bm25_params(100, 10.0).delta, scoring.BM25_PLUS_DELTA)
        with env(RETRIEVER_SCORER="bm25"):
            self.assertEqual(scoring.bm25_params(100, 10.0).delta, 0.0)

    def test_bm25plus_adds_idf_times_delta_when_term_present(self) -> None:
        plain = scoring.Bm25Params(n_docs=100, avg_len=10.0)
        plus = scoring.Bm25Params(n_docs=100, avg_len=10.0, delta=1.0)
        for doc_len in (3, 10, 40):
            with self.subTest(doc_len=doc_len):
                self.assertAlmostEqual(plus.weight(1, doc_len, 5), plain.weight(1, doc_len, 5) + plain.idf(5))
        self.assertEqual(plus.weight(0, 10, 5), 0.0)

    def test_bm25plus_narrows_the_long_chunk_gap(self) -> None:
        plain = scoring.Bm25Params(n_docs=100, avg_len=10.0)
        plus = scoring.Bm25Params(n_docs=100, avg_len=10.0, delta=1.0)
        gap = lambda p: p.weight(1, 3, 5) / p.weight(1, 30, 5)  # noqa: E731
        self.assertLess(gap(plus), gap(plain))

    def test_k1_env_override_and_fallback(self) -> None:
        with env(RETRIEVER_BM25_K1="1.2"):
            self.assertEqual(scoring.bm25_k1(), 1.2)
        with env(RETRIEVER_BM25_K1="abc"):
            self.assertEqual(scoring.bm25_k1(), scoring.K1)


class FakeScorer:
    """본문에 '정답'이 들어 있으면 높은 점수. 호출 횟수를 센다."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.calls += 1
        return [1.0 if "정답" in p else 0.0 for p in passages]


class RerankTests(CorpusCase):
    def setUp(self) -> None:
        super().setUp()
        from org_agent_mvp import rerank
        self.rerank = rerank
        self.fake = FakeScorer()
        rerank._LOADED[("fake-model", rerank.DEFAULT_MAX_LENGTH)] = self.fake
        self.addCleanup(rerank._LOADED.pop, ("fake-model", rerank.DEFAULT_MAX_LENGTH), None)

    rows = [
        chunk("a", "보고서 가", "이월 이월 이월 이월 금액", index=1, page_nos=[1]),
        chunk("b", "보고서 나", "이월 이월 사유", index=1, page_nos=[2]),
        chunk("c", "보고서 다", "이월 정답 문단", index=1, page_nos=[3]),
    ]

    def test_default_is_off(self) -> None:
        with env(RETRIEVER_RERANK="none"):
            self.assertIsNone(self.rerank.get_scorer())

    def test_reorder_keeps_score_values_and_changes_order(self) -> None:
        corpus = self.load(self.rows)
        with env(RETRIEVER_RERANK="none"):
            base = corpus.search_chunks(["이월"], query_text="이월", top_k=5)
        with env(RETRIEVER_RERANK="fake-model", RETRIEVER_RERANK_TOP_N="3"):
            reranked = corpus.search_chunks(["이월"], query_text="이월", top_k=5)
        self.assertEqual(sorted(s for s, _ in base), sorted(s for s, _ in reranked))   # 점수 값 모음은 같다
        self.assertNotEqual(base[0][1]["document_id"], "c")
        self.assertEqual(reranked[0][1]["document_id"], "c")                           # 순서만 바뀐다
        self.assertEqual(reranked[0][0], base[0][0])                                    # 1위 점수 값은 원래 1위 값

    def test_only_top_n_is_reordered(self) -> None:
        corpus = self.load(self.rows)
        with env(RETRIEVER_RERANK="fake-model", RETRIEVER_RERANK_TOP_N="2"):
            reranked = corpus.search_chunks(["이월"], query_text="이월", top_k=5)
        self.assertEqual(reranked[-1][1]["document_id"], "c")   # 상위 2개 밖이라 그대로 3위

    def test_document_search_follows_reranked_order(self) -> None:
        corpus = self.load(self.rows)
        with env(RETRIEVER_RERANK="fake-model", RETRIEVER_RERANK_TOP_N="3"):
            _, card = corpus.search(["이월"], query_text="이월", top_k=1)[0]
        self.assertEqual(card["source_ref"]["document_id"], "c")


class DocMetaSwitchTests(CorpusCase):
    def test_ltm_doc_meta_none_ignores_sidecar(self) -> None:
        rows = [chunk("a", "협약서", "총사업비")]
        (Path(self._tmp.name) / "document_meta.jsonl").write_text(
            json.dumps({"doc_id": "a", "doc_type": "agreement"}) + "\n", encoding="utf-8")
        self.assertEqual(self.load(rows).source_types(), ["agreement"])
        with env(LTM_DOC_META="none"):
            self.assertEqual(self.load(rows).source_types(), ["document"])


if __name__ == "__main__":
    unittest.main()
