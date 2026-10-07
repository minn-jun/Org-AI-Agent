"""측정 스크립트의 채점 정규화 `loose()` — 판정 오탐 회귀 (2026-10-07).

이 함수는 지표를 직접 바꾼다. 같은 값을 다르게 적었을 뿐인데 "오답"으로
세면, 고쳐야 할 것을 찾지 못하고 엉뚱한 손잡이를 돌리게 된다. 실제로
두 번 샜다.

    1차  진행 과제 20문항에서 3건 오탐 — PDF 공백 · 구분자 · 천 단위 콤마
    2차  연쇄 27턴에서 2건 오탐 — 대소문자(`Jev`) · 한글 날짜(`2026년 4월 1일`)

반대 방향도 고정한다. 느슨하게 만들수록 **서로 다른 값을 같다고 하는**
위험이 커진다. 날짜 자리를 0으로 채우는 것이 그 경계다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from answer_check_prentice import loose  # noqa: E402


class LooseMatchTests(unittest.TestCase):
    def assert_same(self, expected: str, answer: str) -> None:
        self.assertIn(loose(expected), loose(answer), f"{expected!r} vs {answer!r}")

    def assert_different(self, expected: str, answer: str) -> None:
        self.assertNotIn(loose(expected), loose(answer), f"{expected!r} vs {answer!r}")

    # ── 1차 오탐 (진행 과제 20문항)
    def test_pdf_extraction_space_inside_number(self):
        self.assert_same("12345678", "과제번호는 RS-2026-1234567 8 입니다")

    def test_date_separator(self):
        self.assert_same("2030-12-15", "협약 종료일은 2030.12.15입니다")

    def test_thousands_comma(self):
        self.assert_same("111000", "정부지원연구개발비는 111,000천원입니다")

    # ── 2차 오탐 (연쇄 27턴)
    def test_letter_case(self):
        self.assert_same("jev", "**Jev 같은 모델**을 활용하라는 피드백이었어")

    def test_korean_full_date(self):
        self.assert_same("2026-04-01", "2026년 4월 1일부터 시작합니다")
        self.assert_same("2027-09-30", "2027년 9월 30일까지입니다")

    def test_korean_month_day_only(self):
        self.assert_same("08-27", "2026년 8월 27일 세미나에서 나왔습니다")
        self.assert_same("09-18", "9월 18일에 보고했습니다")

    def test_expected_value_must_be_written_as_mm_dd(self):
        """기대값에 `8월 27`처럼 `일`이 빠진 형태를 쓰면 안 만난다.

        정규화기를 `일` 없이도 걸리게 만들면 "지난 8월 2건" 같은 말을
        `08-02`로 읽는다. 느슨하게 하지 않고 평가셋을 `MM-DD`로 통일했다.
        이 테스트는 그 약속을 적어 둔다.
        """
        self.assert_different("8월 27", "8월 27일 세미나")
        self.assert_same("08-27", "8월 27일 세미나")

    def test_korean_date_is_zero_padded(self):
        """`4월 1일`이 `0401`이 되어야 `2026-04-01`과 만난다."""
        self.assertIn("20260401", loose("2026년 4월 1일"))
        self.assertNotIn("202641", loose("2026년 4월 1일"))

    # ── 느슨해도 넘지 말아야 할 선
    def test_different_year_stays_different(self):
        self.assert_different("2026-04-01", "2027년 4월 1일에 시작합니다")

    def test_different_amount_stays_different(self):
        self.assert_different("222800", "총 333,800천원입니다")

    def test_different_patent_stays_different(self):
        self.assert_different("10-1111111", "특허 10-2222222을 보유하고 있습니다")

    def test_month_day_does_not_swallow_a_longer_number(self):
        """`12월 15일`은 `1215`다. `121`이나 `215`로 흘러선 안 된다."""
        self.assertIn("1215", loose("12월 15일"))
        self.assert_different("2026-12-15", "12월 15일에 종료합니다")


if __name__ == "__main__":
    unittest.main()
