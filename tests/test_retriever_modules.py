"""검색기 교체 모듈(tokenizer / scoring / dense / ltm_corpus)의 회귀 테스트.

이 네 모듈은 검색 결과를 직접 결정하는데 단위 테스트가 없었다.
평가 스크립트로만 확인하면 수치가 흔들렸을 때 원인이 코퍼스인지 코드인지
갈라지지 않는다. 코퍼스 없이도 도는 검사를 여기에 둔다.

LTM 코퍼스는 저장소 밖에 있으므로, 실제 파일 대신 같은 스키마의 작은
chunks.jsonl을 임시로 만들어 쓴다. 스키마가 바뀌면 여기서 먼저 깨진다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from org_agent_mvp import dense as dense_mod
from org_agent_mvp import scoring, tokenizer
from org_agent_mvp.ltm_corpus import DEFAULT_DOC_TYPE, LtmCorpus
from org_agent_mvp.memory_store import MemoryStore
from org_agent_mvp.prefetch import MemoryPrefetcher, prefetch_query
from org_agent_mvp.query_analyzer import QueryPlan

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def env(**values: str):
    """환경변수를 임시로 바꾼다. 검색기 단계가 전부 환경변수로 갈린다."""
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


# --------------------------------------------------------------------- tokenizer


class TokenizerTests(unittest.TestCase):
    def test_default_is_morph(self) -> None:
        """2026-09-16 기본값이 형태소다. kiwipiepy가 없는 환경에서는 공백 분리로 물러난다."""
        expected = "morph" if tokenizer.available()["morph"] else "whitespace"
        with env(RETRIEVER_TOKENIZER=""):
            os.environ.pop("RETRIEVER_TOKENIZER")
            self.assertEqual(tokenizer.tokenizer_name(), expected)

    def test_unknown_name_falls_back_to_default(self) -> None:
        expected = "morph" if tokenizer.available()["morph"] else "whitespace"
        with env(RETRIEVER_TOKENIZER="bogus"):
            self.assertEqual(tokenizer.tokenizer_name(), expected)

    def test_whitespace_can_still_be_selected(self) -> None:
        with env(RETRIEVER_TOKENIZER="whitespace"):
            self.assertEqual(tokenizer.tokenizer_name(), "whitespace")

    def test_whitespace_splits_on_character_class(self) -> None:
        self.assertEqual(
            tokenizer.whitespace_tokenize("예산 15,955천원 SM-LowV-001"),
            ["예산", "15", "955천원", "sm", "lowv", "001"],
        )

    def test_whitespace_lowercases(self) -> None:
        self.assertEqual(tokenizer.whitespace_tokenize("WebRTC"), ["webrtc"])

    def test_tokenize_many_matches_tokenize_one_by_one(self) -> None:
        """배치와 단건이 다르면 색인과 질의가 어긋난다. 모듈 전체의 전제다."""
        texts = [
            "연구개발과제의 총사업비는 1,346,270천원이다",
            "WebRTC를 활용하여 영상/음성 정보를 송수신한다",
            "",
        ]
        for name in ("whitespace", "morph"):
            if not tokenizer.available()[name]:
                continue
            with self.subTest(tokenizer=name), env(RETRIEVER_TOKENIZER=name):
                self.assertEqual(
                    tokenizer.tokenize_many(texts),
                    [tokenizer.tokenize(text) for text in texts],
                )

    def test_empty_text_yields_no_tokens(self) -> None:
        for name in ("whitespace", "morph"):
            if not tokenizer.available()[name]:
                continue
            with self.subTest(tokenizer=name), env(RETRIEVER_TOKENIZER=name):
                self.assertEqual(tokenizer.tokenize(""), [])

    @unittest.skipUnless(tokenizer.available()["morph"], "kiwipiepy가 없다")
    def test_morph_strips_particles(self) -> None:
        """조사가 붙은 질의어와 문서의 표면형이 만나야 한다. 1단계의 존재 이유다."""
        self.assertIn("인건비", tokenizer.morph_tokenize("인건비를 연구활동비로 바꿨다"))

    @unittest.skipUnless(tokenizer.available()["morph"], "kiwipiepy가 없다")
    def test_morph_drops_contentless_forms(self) -> None:
        tokens = tokenizer.morph_tokenize("그렇게 할 수 있는 것 등")
        self.assertNotIn("수", tokens)
        self.assertNotIn("것", tokens)


# ----------------------------------------------------------------------- scoring


class ScoringTests(unittest.TestCase):
    def test_default_scorer_is_bm25_for_both_scopes(self) -> None:
        """2026-09-16 기본값이 BM25다. 계층마다 다르면 눈금이 벌어져 병합이 무너진다(07 문서)."""
        with env(RETRIEVER_SCORER="", RETRIEVER_SCORER_SEED=""):
            os.environ.pop("RETRIEVER_SCORER")
            os.environ.pop("RETRIEVER_SCORER_SEED")
            self.assertEqual(scoring.scorer_name(), "bm25")
            self.assertEqual(scoring.scorer_name("seed"), "bm25")

    def test_seed_scope_can_differ_from_ltm(self) -> None:
        """코퍼스 크기가 100배 차이나서 계층별로 다른 점수 함수를 줄 수 있어야 한다."""
        with env(RETRIEVER_SCORER="bm25", RETRIEVER_SCORER_SEED="freq"):
            self.assertEqual(scoring.scorer_name(), "bm25")
            self.assertEqual(scoring.scorer_name("seed"), "freq")

    def test_seed_scope_follows_base_when_unset(self) -> None:
        with env(RETRIEVER_SCORER="bm25", RETRIEVER_SCORER_SEED=""):
            os.environ.pop("RETRIEVER_SCORER_SEED")
            self.assertEqual(scoring.scorer_name("seed"), "bm25")

    def test_unknown_scorer_falls_back(self) -> None:
        with env(RETRIEVER_SCORER="bogus"):
            self.assertEqual(scoring.scorer_name(), "bm25")

    def test_freq_weight_grows_with_frequency_and_saturates(self) -> None:
        self.assertLess(scoring.freq_weight(1), scoring.freq_weight(5))
        # 로그라 증가 폭이 줄어든다.
        self.assertLess(
            scoring.freq_weight(10) - scoring.freq_weight(5),
            scoring.freq_weight(5) - scoring.freq_weight(1),
        )

    def test_bm25_idf_shrinks_as_term_gets_common(self) -> None:
        params = scoring.Bm25Params(n_docs=1000, avg_len=100.0)
        self.assertGreater(params.idf(df=1), params.idf(df=100))
        self.assertGreater(params.idf(df=100), params.idf(df=900))

    def test_bm25_penalises_long_documents(self) -> None:
        """같은 빈도면 긴 문서가 낮아야 한다. 길이 보정이 하는 일이다."""
        params = scoring.Bm25Params(n_docs=1000, avg_len=100.0)
        short = params.weight(tf=3, doc_len=50, df=10)
        long = params.weight(tf=3, doc_len=500, df=10)
        self.assertGreater(short, long)

    def test_bm25_weight_grows_with_frequency(self) -> None:
        params = scoring.Bm25Params(n_docs=1000, avg_len=100.0)
        self.assertLess(
            params.weight(tf=1, doc_len=100, df=10),
            params.weight(tf=5, doc_len=100, df=10),
        )

    def test_bm25_params_guard_degenerate_corpus(self) -> None:
        """빈 코퍼스에서도 0으로 나누지 않는다."""
        params = scoring.Bm25Params(n_docs=0, avg_len=0.0)
        self.assertGreaterEqual(params.weight(tf=1, doc_len=1, df=1), 0.0)

    def test_normalize_mode_default_is_global(self) -> None:
        with env(PREFETCH_NORMALIZE=""):
            os.environ.pop("PREFETCH_NORMALIZE")
            self.assertEqual(scoring.normalize_mode(), "global")

    def test_normalize_mode_rejects_unknown(self) -> None:
        with env(PREFETCH_NORMALIZE="bogus"):
            self.assertEqual(scoring.normalize_mode(), "global")


# ------------------------------------------------------------------------- dense


class DenseTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        with env(RETRIEVER_DENSE=""):
            os.environ.pop("RETRIEVER_DENSE")
            self.assertFalse(dense_mod.enabled())

    def test_enabled_accepts_common_truthy_spellings(self) -> None:
        for value in ("1", "true", "yes"):
            with self.subTest(value=value), env(RETRIEVER_DENSE=value):
                self.assertTrue(dense_mod.enabled())

    def test_weight_falls_back_on_garbage(self) -> None:
        with env(RETRIEVER_DENSE_WEIGHT="not-a-number"):
            self.assertEqual(dense_mod.dense_weight(), 1.0)

    def test_rrf_prefers_documents_found_by_both(self) -> None:
        """양쪽에서 잡힌 청크가 한쪽에서만 잡힌 청크를 이긴다. RRF를 쓰는 이유다.

        `k=60`이라 1~3위 사이의 차이는 거의 평평하다는 점에 주의한다.
        RRF가 실제로 가르는 것은 순위 차이가 아니라 **몇 개의 검색기가 잡았는가**다.
        """
        merged = dense_mod.rrf_merge([1, 2], [1, 3], w_dense=1.0)
        self.assertGreater(merged[1], merged[2])
        self.assertGreater(merged[1], merged[3])

    def test_rrf_weight_controls_dense_influence(self) -> None:
        """w_dense=0이면 sparse 순위만 남아야 한다."""
        merged = dense_mod.rrf_merge([1], [2], w_dense=0.0)
        self.assertGreater(merged[1], 0.0)
        self.assertEqual(merged[2], 0.0)

    def test_rrf_keeps_documents_found_by_only_one_retriever(self) -> None:
        merged = dense_mod.rrf_merge([1], [2], w_dense=0.5)
        self.assertIn(2, merged)


# --------------------------------------------------------------- ltm_corpus 순수 함수


# ------------------------------------------------------------- ltm_corpus 동작


def write_corpus(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk(source_id: str, title: str, text: str, index: int = 1,
          source_path: str = "1단계/04_협약/문서.hwp", **doc_meta) -> dict:
    """청크 한 줄. `doc_meta`는 코퍼스 전처리가 채우는 문서 필드(doc_type, version_group ...)다."""
    return {
        "chunk_id": f"{source_id}-{index:04d}",
        "source_id": source_id,
        "chunk_index": index,
        "chunk_count": 1,
        "metadata": {
            "title": title,
            "source_path": source_path,
            "modified_at": "2026-08-26T11:35:25",
            "stage": "1단계",
            **doc_meta,
        },
        "text": text,
    }


class LtmCorpusTests(unittest.TestCase):
    """실제 코퍼스 없이 도는 검사. 스키마와 계약을 고정한다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "chunks.jsonl"
        # 질의를 미리 잘라 넣는 검사들이라 토큰화를 0단계로 고정한다.
        # 기본값(형태소)은 `총사업비`를 `총`+`사업비`로 쪼개서 이 질의들이 안 맞는다.
        pin = env(RETRIEVER_TOKENIZER="whitespace")
        pin.__enter__()
        self.addCleanup(pin.__exit__, None, None, None)

    def load(self, rows: list[dict]) -> LtmCorpus:
        write_corpus(self.path, rows)
        return LtmCorpus(self.path)

    def test_search_returns_document_level_cards(self) -> None:
        corpus = self.load([
            chunk("doc1", "이월 승인 요청서", "이월 신청액은 15,955천원이다", index=1),
            chunk("doc1", "이월 승인 요청서", "사유는 신규 채용 지연이다", index=2),
        ])
        results = corpus.search(["이월"], top_k=5)
        # 청크 두 개가 걸려도 근거는 문서 하나다.
        self.assertEqual(len(results), 1)
        _, card = results[0]
        self.assertEqual(card["evidence_id"], "ev_ltm_doc1")
        self.assertEqual(card["tier"], "LTM")
        self.assertEqual(card["source_ref"]["document_id"], "doc1")

    def test_evidence_id_is_document_id_not_title(self) -> None:
        """제목은 698문서 중 79개가 중복이라 근거 id로 쓸 수 없다."""
        corpus = self.load([
            chunk("aaa", "확인서", "가천대 확인서 내용"),
            chunk("bbb", "확인서", "산기평 확인서 내용"),
        ])
        results = corpus.search(["확인서"], top_k=5)
        ids = {card["evidence_id"] for _, card in results}
        self.assertEqual(len(ids), len(results))

    def test_empty_query_returns_nothing(self) -> None:
        corpus = self.load([chunk("doc1", "계획서", "본문")])
        self.assertEqual(corpus.search([], top_k=5), [])

    def test_unknown_token_returns_nothing(self) -> None:
        corpus = self.load([chunk("doc1", "계획서", "본문")])
        self.assertEqual(corpus.search(["존재하지않는토큰zzz"], top_k=5), [])

    def test_versions_are_folded_into_one_card(self) -> None:
        corpus = self.load([
            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원", version_group="차단계 사업계획서"),
            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원", version_group="차단계 사업계획서"),
        ])
        results = corpus.search(["총사업비"], top_k=5)
        self.assertEqual(len(results), 1)
        _, card = results[0]
        self.assertEqual(card["source_ref"]["folded_count"], 1)

    def test_folded_ids_are_kept_for_gold_labels(self) -> None:
        """gold가 어느 버전을 가리키든 맞출 수 있어야 평가가 성립한다."""
        corpus = self.load([
            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원", version_group="차단계 사업계획서"),
            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원", version_group="차단계 사업계획서"),
        ])
        _, card = corpus.search(["총사업비"], top_k=5)[0]
        everything = {card["source_ref"]["document_id"], *card["source_ref"]["folded_document_ids"]}
        self.assertEqual(everything, {"v3", "v9"})

    def test_representative_is_the_newest_version(self) -> None:
        """묶음 대표는 점수 1등이 아니라 최신본이다. 사람이 읽어야 할 건 최신본이다."""
        corpus = self.load([
            chunk("v3", "차단계 사업계획서_v3", "총사업비 총사업비 총사업비", version_group="차단계 사업계획서", version_rank=[0, 3, 0]),
            chunk("v9", "차단계 사업계획서_v9", "총사업비 한 번", version_group="차단계 사업계획서", version_rank=[0, 9, 0]),
        ])
        _, card = corpus.search(["총사업비"], top_k=5)[0]
        self.assertEqual(card["source_ref"]["document_id"], "v9")

    def test_unrelated_documents_are_not_folded_together(self) -> None:
        corpus = self.load([
            chunk("a", "협약 변경 신청서", "연구개발기간 을 변경 한다"),
            chunk("b", "연구비 정산 보고서", "연구개발기간 이 끝났다"),
        ])
        results = corpus.search(["연구개발기간"], top_k=5)
        self.assertEqual(len(results), 2)

    def test_identical_bodies_fold_even_across_families(self) -> None:
        """제목 계열이 달라도 대표 청크 본문이 같으면 한 묶음이다.

        같은 문단이 여러 파일에 복사된 경우가 실제로 많아서, 계열 키만으로는
        중복이 남는다. 접기 기준이 계열 키 **또는** 본문 해시인 이유다.
        """
        corpus = self.load([
            chunk("a", "협약 변경 신청서", "연구개발기간 변경"),
            chunk("b", "연구비 정산 보고서", "연구개발기간 변경"),
        ])
        self.assertEqual(len(corpus.search(["연구개발기간"], top_k=5)), 1)

    def test_project_filter_penalises_instead_of_dropping(self) -> None:
        """하드 필터는 analyzer 오판 하나로 후보를 0건으로 만든다."""
        corpus = self.load([chunk("doc1", "계획서", "총사업비 내용")])
        kept = corpus.search(["총사업비"], filters={"project": "다른 과제"},
                             filter_penalty=0.3, top_k=5)
        self.assertEqual(len(kept), 1)
        base = corpus.search(["총사업비"], top_k=5)
        self.assertLess(kept[0][0], base[0][0])

    def test_zero_penalty_behaves_as_hard_filter(self) -> None:
        corpus = self.load([chunk("doc1", "계획서", "총사업비 내용")])
        self.assertEqual(
            corpus.search(["총사업비"], filters={"project": "다른 과제"},
                          filter_penalty=0.0, top_k=5),
            [],
        )

    def test_title_match_outranks_body_only_match_when_bonus_on(self) -> None:
        """가산점은 2026-09-16부터 기본 꺼짐이다. 켠 경우의 동작만 여기서 고정한다."""
        rows = [
            chunk("titled", "이월 승인 요청서", "무관한 본문"),
            chunk("body", "다른 문서", "이월"),
        ]
        with env(RETRIEVER_TITLE_BONUS="add"):
            results = self.load(rows).search(["이월"], top_k=5)
        self.assertEqual(results[0][1]["source_ref"]["document_id"], "titled")

    def test_title_match_does_not_outrank_by_default(self) -> None:
        corpus = self.load([
            chunk("titled", "이월 승인 요청서", "무관한 본문"),
            chunk("body", "다른 문서", "이월"),
        ])
        self.assertEqual(corpus.search(["이월"], top_k=5)[0][1]["source_ref"]["document_id"], "body")

    def test_boilerplate_across_many_families_is_damped(self) -> None:
        """4개 이상 계열에 같은 문단이 나오면 그건 내용이 아니라 서식이다."""
        form = "연구시설 장비명 규격 구입단가 구입연도"
        rows = [chunk(f"form{i}", f"과업지시서 {i}번 문서", form) for i in range(6)]
        rows.append(chunk("real", "장비 구입 계획", form))
        corpus = self.load(rows)
        weights = corpus._chunk_weight
        self.assertTrue(all(w < 1.0 for w in weights))

    def test_content_shared_by_few_families_is_not_damped(self) -> None:
        shared = "시각장애인 인구 전망 자료"
        corpus = self.load([
            chunk("a", "미정리 조사 자료", shared),
            chunk("b", "그림자료 모음집", shared),
        ])
        self.assertTrue(all(w == 1.0 for w in corpus._chunk_weight))

    def test_stats_report_index_size(self) -> None:
        corpus = self.load([
            chunk("a", "문서 하나", "본문 하나", index=1),
            chunk("a", "문서 하나", "본문 둘", index=2),
            chunk("b", "문서 둘", "본문 셋"),
        ])
        stats = corpus.stats()
        self.assertEqual(stats["documents"], 2)
        self.assertEqual(stats["chunks"], 3)
        self.assertGreater(stats["vocabulary"], 0)

    def test_source_types_come_from_document_meta(self) -> None:
        corpus = self.load([
            chunk("a", "협약서", "본문", doc_type="agreement"),
            chunk("b", "지시서", "본문", doc_type="patent_document"),
        ])
        self.assertEqual(corpus.source_types(), ["agreement", "patent_document"])

    def test_retriever_knows_no_corpus_rules(self) -> None:
        """메타데이터가 없으면 폴더 이름·파일명 꼬리표로 추정하지 않는다.

        2026-09-15에 20200504 과제 폴더 전용 규칙을 코퍼스 전처리로 옮겼다.
        폴더가 "04_협약"이어도 doc_type이 없으면 기본값이고, "_v3"/"_v9"도 접지 않는다.
        """
        corpus = self.load([
            chunk("v3", "차단계 사업계획서_v3", "총사업비 1,346,270천원", source_path="1단계/04_협약/a.hwp"),
            chunk("v9", "차단계 사업계획서_v9", "총사업비 1,346,270천원 변경", source_path="06_최종제출/b.hwp"),
        ])
        self.assertEqual(corpus.source_types(), [DEFAULT_DOC_TYPE])
        self.assertEqual(corpus.projects(), [])
        self.assertEqual(len(corpus.search(["총사업비"], top_k=5)), 2)

    def test_source_type_is_never_a_file_format(self) -> None:
        """source_type이 pdf/hwp가 되면 analyzer enum이 형식 이름으로 오염된다."""
        corpus = self.load([chunk("a", "문서", "본문", source_path="a/b.pdf")])
        self.assertNotIn(corpus.source_types()[0], {"pdf", "hwp", "pptx"})

    def test_final_document_beats_higher_version_as_representative(self) -> None:
        """is_final(실제 제출본)은 파일명 버전 번호보다 강한 신호다."""
        corpus = self.load([
            chunk("v4", "신청용 계획서_v4", "총사업비 내용", version_group="신청용 계획서", version_rank=[0, 4, 0]),
            chunk("sub", "신청용 계획서", "총사업비 내용 제출", version_group="신청용 계획서", is_final=True),
        ])
        _, card = corpus.search(["총사업비"], top_k=5)[0]
        self.assertEqual(card["source_ref"]["document_id"], "sub")

    def test_document_meta_file_overrides_chunk_metadata(self) -> None:
        """문서 필드는 chunks.jsonl 옆 document_meta.jsonl에 둘 수 있고, 청크 metadata보다 우선한다."""
        write_corpus(self.path, [chunk("a", "협약서", "총사업비", doc_type="agreement", project="A 과제")])
        (self.path.parent / "document_meta.jsonl").write_text(
            json.dumps({"doc_id": "a", "doc_type": "official_report", "project": "B 과제"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        corpus = LtmCorpus(self.path)
        self.assertEqual(corpus.source_types(), ["official_report"])
        self.assertEqual(corpus.projects(), ["B 과제"])
        _, card = corpus.search(["총사업비"], top_k=1)[0]
        self.assertEqual(card["project"], "B 과제")

    def test_project_filter_uses_each_documents_project(self) -> None:
        corpus = self.load([
            chunk("a", "계획서 가", "총사업비 내용", project="A 과제"),
            chunk("b", "계획서 나", "총사업비 내용 추가", project="B 과제"),
        ])
        results = corpus.search(["총사업비"], filters={"project": "A 과제"}, filter_penalty=0.0, top_k=5)
        self.assertEqual([card["source_ref"]["document_id"] for _, card in results], ["a"])

    def test_card_shape_matches_memory_store_contract(self) -> None:
        """prefetch가 두 계층을 한 목록에 섞으므로 카드 모양이 같아야 한다."""
        corpus = self.load([chunk("doc1", "계획서", "총사업비 내용")])
        _, card = corpus.search(["총사업비"], top_k=1)[0]
        for key in ("evidence_id", "tier", "source_type", "title", "date",
                    "project", "summary", "quote", "content_excerpt",
                    "source_ref", "retrieval_score", "permission_scope"):
            self.assertIn(key, card)

    def test_search_is_deterministic(self) -> None:
        """실행마다 순서가 흔들리면 gold 라벨이 붙었다 떨어졌다 한다."""
        rows = [chunk(f"d{i}", f"문서 {i}번 제목", "총사업비 내용") for i in range(10)]
        corpus = self.load(rows)
        first = [card["evidence_id"] for _, card in corpus.search(["총사업비"], top_k=10)]
        second = [card["evidence_id"] for _, card in corpus.search(["총사업비"], top_k=10)]
        self.assertEqual(first, second)

    def test_query_token_order_does_not_change_which_documents_are_found(self) -> None:
        """질의 토큰 순서가 바뀌어도 **어떤 문서가 몇 점으로** 걸리는지는 같아야 한다.

        정확히 동점인 문서들 사이의 순서까지 같으라고 요구하지는 않는다.
        그건 명시적인 동점 규칙을 새로 세워야 얻어지는데, 실측해 보니 그 규칙이
        계열 접기의 대표 청크를 바꿔서 결과가 실제로 달라졌다
        (`ltm-webrtc` 케이스의 정답이 8위에서 9위로 밀렸다).
        재현성에 필요한 것은 여기까지이고, 질의 토큰 순서 자체는 토크나이저가
        정해 주므로 같은 질의에서 흔들리지 않는다.
        """
        rows = [
            chunk("a", "협약 변경 신청서 하나", "연구개발기간 총사업비 변경 신청"),
            chunk("b", "연구비 정산 보고서 둘", "총사업비 연구개발기간 정산 결과"),
            chunk("c", "이월 승인 요청서 셋", "총사업비 이월 연구개발기간 승인"),
        ]
        corpus = self.load(rows)
        tokens = ["총사업비", "연구개발기간", "신청", "정산", "이월"]

        def scored(order: list[str]) -> list[tuple[str, float]]:
            found = [(card["evidence_id"], round(score, 6))
                     for score, card in corpus.search(order, top_k=10)]
            return sorted(found)

        expected = scored(tokens)
        for shift in range(1, len(tokens)):
            rotated = tokens[shift:] + tokens[:shift]
            with self.subTest(order=rotated):
                self.assertEqual(scored(rotated), expected)

    def test_search_is_deterministic_across_processes(self) -> None:
        """해시 시드가 달라도 같은 결과가 나와야 한다.

        같은 프로세스 안에서만 검사하면 이 결함을 못 잡는다. 문자열 해시는
        프로세스 시작 때 한 번 정해지므로, 한 프로세스 안에서는 set의 순회
        순서도 고정되기 때문이다. 실제로 실코퍼스 60건에서 같은 명령의
        재현율이 62.0%와 62.9% 사이를 오갔다.
        """
        # 동점을 일부러 만든다. docA의 두 청크는 점수가 정확히 같은데
        # 본문이 다르다. 어느 쪽이 대표가 되느냐에 따라 docB와 본문 해시가
        # 같아져 한 건으로 접히기도 하고, 두 건으로 남기도 한다.
        #
        # 질의어가 둘이라, 어느 토큰의 posting을 먼저 훑느냐가 대표를 정한다.
        # 그 순서를 set이 정하던 때는 해시 시드에 따라 결과가 1건과 2건 사이를
        # 오갔다(시드 0/2/4에서 2건, 1/3에서 1건).
        rows = [
            chunk("docA", "협약 변경 신청서 하나", "알파", index=1),
            chunk("docA", "협약 변경 신청서 하나", "베타", index=2),
            chunk("docB", "연구비 정산 보고서 둘", "알파", index=1),
        ]
        write_corpus(self.path, rows)
        script = (
            "import json,sys;"
            "sys.path.insert(0, sys.argv[2]);"
            "from org_agent_mvp.ltm_corpus import LtmCorpus;"
            "c=LtmCorpus(sys.argv[1]);"
            "print(json.dumps([k['evidence_id'] for _,k in "
            "c.search(['알파','베타'], top_k=12)]))"
        )
        outputs = set()
        for seed in ("0", "1", "2", "3", "4"):
            environ = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8"}
            completed = subprocess.run(
                [sys.executable, "-c", script, str(self.path), str(PROJECT_ROOT)],
                capture_output=True, text=True, encoding="utf-8", env=environ,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            outputs.add(completed.stdout.strip())
        self.assertEqual(len(outputs), 1, f"해시 시드마다 결과가 달랐다: {outputs}")


# ------------------------------------------------------- 이번에 고친 결함의 회귀 검사


class QueryTokenConsistencyTests(unittest.TestCase):
    """질의 토큰이 계층마다 다르게 세어지면 계층 병합이 어긋난다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        (root / "stm").mkdir()
        (root / "stm" / "note.md").write_text(
            "---\ntitle: 예산 메모\nproject: 공통\n---\n예산 단가 산정 근거\n",
            encoding="utf-8",
        )
        self.store = MemoryStore(root)

    def test_expanded_query_tokens_are_deduplicated(self) -> None:
        """QUERY_EXPANSIONS의 값이 겹쳐 같은 토큰이 여러 번 들어온다.

        중복을 그대로 두면 그 토큰의 배점이 배수가 되는데, 근거는 동의어 표가
        어쩌다 그렇게 적혀 있다는 것뿐이다. LTM은 set()으로 훑어 중복을 세지
        않으므로 계층마다 눈금이 달라지기도 한다.
        """
        with env(QUERY_EXPANSIONS="default"):
            tokens, _ = self.store._prepare_query("예산 비용 단가 기준")
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_expansion_still_adds_synonyms(self) -> None:
        """중복만 없앤다. 확장 자체가 사라지면 안 된다. (사전은 2026-09-16부터 기본 꺼짐)"""
        with env(QUERY_EXPANSIONS="default"):
            tokens, _ = self.store._prepare_query("예산")
        self.assertIn("사업비", tokens)

    def test_expansion_is_off_by_default(self) -> None:
        with env(QUERY_EXPANSIONS=""):
            os.environ.pop("QUERY_EXPANSIONS")
            tokens, _ = self.store._prepare_query("예산")
        self.assertNotIn("사업비", tokens)


class PrefetchQueryTests(unittest.TestCase):
    """검색 질의를 만드는 곳이 여러 군데라 서로 달랐다. 한 함수로 모았다."""

    @staticmethod
    def plan(rewrites: list[str]) -> QueryPlan:
        return QueryPlan(
            intent="organization_memory_lookup",
            can_answer_directly=False,
            memory_needed=True,
            answer_source="memory_prefetch",
            use_session_context=False,
            memory_weights={"stm": 0.34, "mtm": 0.43, "ltm": 0.23},
            query_rewrites=rewrites,
            filters={},
            reason="",
        )

    def test_all_rewrites_are_used(self) -> None:
        """마지막 rewrite만 쓰면 원 질문의 정보가 빠진다."""
        query = prefetch_query(self.plan(["원 질문", "이전 질문: X 후속 질문: 원 질문"]))
        self.assertIn("원 질문", query)
        self.assertIn("이전 질문", query)

    def test_duplicate_rewrites_are_collapsed(self) -> None:
        self.assertEqual(prefetch_query(self.plan(["같은 질문", "같은 질문"])), "같은 질문")

    def test_single_rewrite_is_unchanged(self) -> None:
        self.assertEqual(prefetch_query(self.plan(["질문 하나"])), "질문 하나")


# ------------------------------------------------- 계층 병합 정규화 (zscore / hybrid)


class _StubStore:
    """계층마다 정해진 점수를 돌려주는 가짜 저장소.

    실제 코퍼스 없이 **눈금 격차만** 재현한다. LTM은 BM25라 점수가 크고
    STM은 빈도라 작다 — 실측한 2.44배 격차를 그대로 흉내낸다.
    """

    def __init__(self, per_tier: dict[str, list[float]]):
        self.per_tier = per_tier

    def retrieve(self, *, tier, query, filters, top_k):
        scores = self.per_tier.get(tier, [])[:top_k]
        return {
            "result_count": len(scores),
            "results": [
                {"evidence_id": f"ev_{tier}_{i}", "tier": tier.upper(),
                 "retrieval_score": s}
                for i, s in enumerate(scores)
            ],
        }


def _plan(weights: dict[str, float]) -> QueryPlan:
    return QueryPlan(
        intent="organization_memory_lookup", can_answer_directly=False,
        memory_needed=True, answer_source="memory_prefetch",
        use_session_context=False, memory_weights=weights,
        query_rewrites=["질문"], filters={}, reason="",
    )


class NormalizeModeTests(unittest.TestCase):
    """계층 병합에서 점수 눈금을 어떻게 맞출지."""

    #: LTM만 점수가 2.4배 크다. 그런데 STM에는 볼 만한 것이 하나도 없다
    #: (전부 낮고 서로 비슷하다). 이 상황에서 STM을 띄우면 안 된다.
    SCALES = {
        "ltm": [26.0, 24.0, 20.0, 12.0, 8.0, 6.0, 5.0, 4.0],
        "stm": [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3],
        "mtm": [3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4],
    }
    WEIGHTS = {"stm": 0.34, "mtm": 0.43, "ltm": 0.23}

    def cards(self, mode: str, **extra: str) -> list[dict]:
        with env(PREFETCH_NORMALIZE=mode, **extra):
            pf = MemoryPrefetcher(_StubStore(self.SCALES), total_top_k=8,
                                  alpha=1.0, pool_per_tier=8, tier_floor=0)
            return pf.prefetch(_plan(self.WEIGHTS)).cards

    def top_tier(self, mode: str, **extra: str) -> str:
        return self.cards(mode, **extra)[0]["tier"].lower()

    def test_global_keeps_the_strong_tier_on_top(self) -> None:
        """기준 동작. 점수가 압도적인 계층이 1위를 가져간다."""
        self.assertEqual(self.top_tier("global"), "ltm")

    def test_tier_mode_lets_a_weak_tier_win(self) -> None:
        """계층별 최고점으로 나누면 각 계층 1등이 모두 1.0이 된다.

        볼 것이 없는 STM/MTM의 1등도 LTM의 1등과 같은 점수를 받아서,
        tier_prior가 더 높은 쪽이 이긴다. 실코퍼스에서 LTM MRR이
        0.451 -> 0.176으로 무너진 것이 이 현상이다.
        이 테스트는 고쳐야 할 동작을 **기록**해 둔 것이다.
        """
        self.assertNotEqual(self.top_tier("tier"), "ltm")

    def test_zscore_alone_also_inflates_a_weak_tier(self) -> None:
        """z만 쓰면 tier와 같은 함정에 빠진다.

        풀이 top_k로 잘려 있어서 계층마다 분포 성격이 다르다.
        비어 있는 계층도 자기 풀 안에서는 1등이 튀어 보인다.
        """
        self.assertNotEqual(self.top_tier("zscore"), "ltm")

    def test_hybrid_keeps_the_strong_tier_on_top(self) -> None:
        """global과 기하평균을 내면 절대 품질이 살아난다.

        이것이 hybrid를 넣은 이유다 — 눈금은 맞추되
        '이 계층에 실제로 쓸 만한 것이 있는가'를 버리지 않는다.
        """
        self.assertEqual(self.top_tier("hybrid"), "ltm")

    def test_hybrid_recovers_a_tier_that_global_cuts_away(self) -> None:
        """1위는 안 뺏기되, 약한 계층이 후보로 살아남기는 해야 한다.

        눈금이 2.4배 어긋나면 global에서 STM 최고점이 3.0/26.0 = 0.115로
        눌린다. 상대 컷(최고점의 0.3배)에 걸려 **STM이 통째로 사라진다.**
        계층 혼합 질문이라면 두 번째 계층의 정답을 통째로 놓치는 것이다.

        hybrid는 눈금을 걷어내므로 STM이 후보로 남는다. 그러면서도
        1위는 LTM이 지킨다(위 테스트).
        """
        def tiers(mode: str) -> set[str]:
            return {c["tier"].lower() for c in self.cards(mode)}

        self.assertNotIn("stm", tiers("global"))
        self.assertIn("stm", tiers("hybrid"))

    def test_hybrid_weight_interpolates_between_the_two_ends(self) -> None:
        """w=0이면 global, w=1이면 zscore와 같아야 한다."""
        self.assertEqual(self.top_tier("hybrid", PREFETCH_HYBRID_W="0"),
                         self.top_tier("global"))
        self.assertEqual(self.top_tier("hybrid", PREFETCH_HYBRID_W="1"),
                         self.top_tier("zscore"))

    def test_ranking_inside_a_tier_is_preserved(self) -> None:
        """계층 안에서는 순위가 절대 바뀌면 안 된다.

        z도 hybrid도 계층 안에서는 단조 변환이다. 검색기가 매긴
        순서를 병합이 뒤집으면 그건 정규화가 아니라 재랭킹이다.
        """
        for mode in ("global", "tier", "zscore", "hybrid"):
            with self.subTest(mode=mode):
                cards = self.cards(mode)
                ltm = [c for c in cards if c["tier"].lower() == "ltm"]
                raws = [c["retrieval_score"] for c in ltm]
                norms = [c["normalized_score"] for c in ltm]
                self.assertEqual(raws, sorted(raws, reverse=True))
                self.assertEqual(norms, sorted(norms, reverse=True))

    def test_small_pool_falls_back_to_global(self) -> None:
        """후보가 2건뿐이면 평균·표준편차가 의미 없다.

        그 계층만 global로 처리한다. 예외가 나거나 점수가 지어내지면 안 된다.
        """
        scales = {"ltm": [26.0, 24.0], "stm": [3.0], "mtm": []}
        with env(PREFETCH_NORMALIZE="hybrid"):
            pf = MemoryPrefetcher(_StubStore(scales), total_top_k=8,
                                  alpha=1.0, pool_per_tier=8, tier_floor=0)
            cards = pf.prefetch(_plan(self.WEIGHTS)).cards
        self.assertTrue(cards)
        self.assertEqual(cards[0]["tier"].lower(), "ltm")
        for card in cards:
            self.assertGreaterEqual(card["normalized_score"], 0.0)

    def test_unknown_mode_falls_back_to_default(self) -> None:
        with env(PREFETCH_NORMALIZE="없는모드"):
            self.assertEqual(scoring.normalize_mode(), scoring.DEFAULT_NORMALIZE)

    def test_hybrid_weight_is_clamped(self) -> None:
        for raw, expected in (("-1", 0.0), ("5", 1.0), ("헛소리", scoring.DEFAULT_HYBRID_W)):
            with self.subTest(raw=raw), env(PREFETCH_HYBRID_W=raw):
                self.assertEqual(scoring.hybrid_weight(), expected)


if __name__ == "__main__":
    unittest.main()
