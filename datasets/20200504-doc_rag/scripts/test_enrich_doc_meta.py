"""20200504 전처리 규칙 테스트. 원래 org_agent_mvp/tests/test_retriever_modules.py의 FamilyKeyTests였다.

    python -m unittest test_enrich_doc_meta.py     (이 폴더에서)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from enrich_doc_meta import classify, document_meta, family_key, version_rank  # noqa: E402


class FamilyKeyTests(unittest.TestCase):
    def test_version_tags_are_stripped(self) -> None:
        self.assertEqual(family_key("차단계 사업계획서_v3"), "차단계 사업계획서")
        self.assertEqual(family_key("차단계 사업계획서_v10"), "차단계 사업계획서")

    def test_same_family_across_version_spellings(self) -> None:
        titles = [
            "지식서비스 사업계획서_v7",
            "지식서비스 사업계획서_최종",
            "지식서비스 사업계획서_v7.2_수정본",
        ]
        self.assertEqual(len({family_key(t) for t in titles}), 1)

    def test_short_titles_are_not_stripped_to_nothing(self) -> None:
        """꼬리표를 계속 깎아 제목이 사라지면 무관한 문서가 한 덩어리가 된다.

        가드가 걸리면 꼬리표가 남는다. 접기를 놓치는 대신 오합침을 막는 쪽이다.
        실코퍼스 698건 중 이 가드에 걸리는 것은 5건이다.
        """
        self.assertNotEqual(family_key("보고서_v3"), "")
        self.assertNotEqual(family_key("보고서_v3"), family_key("계획서_v3"))

    def test_empty_title_is_safe(self) -> None:
        self.assertEqual(family_key(""), "")

    def test_version_rank_orders_within_family(self) -> None:
        self.assertLess(version_rank("계획서_v3"), version_rank("계획서_v10"))
        self.assertLess(version_rank("계획서_v7"), version_rank("계획서_v7.2"))

    def test_final_outranks_numbered_drafts(self) -> None:
        """한국어 문서 관행에서 `_최종`은 번호가 매겨진 작업본을 대체한다."""
        self.assertGreater(version_rank("계획서_최종"), version_rank("계획서_v17"))

    def test_version_rank_takes_highest_number_present(self) -> None:
        self.assertEqual(version_rank("계획서_v1_v9")[1], 9)

    def test_classify_maps_folder_to_semantic_type(self) -> None:
        self.assertEqual(classify("1단계/04_협약/협약서.hwp"), "agreement")
        self.assertEqual(classify("2단계/07_특허/과업지시서.hwp"), "patent_document")

    def test_classify_defaults_when_no_rule_matches(self) -> None:
        self.assertEqual(classify(""), "official_document")
        self.assertEqual(classify("알 수 없는 폴더/문서.pdf"), "official_document")

    def test_classify_never_returns_a_file_format(self) -> None:
        """source_type이 pdf/hwp가 되면 analyzer enum이 형식 이름으로 오염된다."""
        for path in ("a/b.pdf", "a/b.hwp", "a/b.pptx"):
            with self.subTest(path=path):
                self.assertNotIn(classify(path), {"pdf", "hwp", "pptx"})

    def test_final_submission_folder_is_marked(self) -> None:
        meta = document_meta("d", "신청용 계획서(PART II)", "1단계/06_최종제출/신청용 계획서(PART II).hwp")
        self.assertTrue(meta["is_final"])
        self.assertFalse(document_meta("d", "계획서_v4", "1단계/02_계획서 작업/계획서_v4.hwp")["is_final"])


if __name__ == "__main__":
    unittest.main()
