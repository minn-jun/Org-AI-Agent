"""내보낸 Codex 대화를 MTM 시드로 바꾸는 변환기를 검사한다.

이 변환기는 새 대화 로그를 받을 때마다 다시 돌린다. 그래서 손으로 고친
결과가 아니라 **규칙이 지켜지는지**를 묶어 둔다. 네 가지가 깨지면 측정이
조용히 틀어진다.

1. 진행 안내(`commentary`)가 근거로 섞여 들어간다.
2. 하루치로 쪼개지지 않아 날짜 질문의 정답 범위가 뭉개진다.
3. 평문 키·메일주소·로컬 경로가 시드에 남는다.
4. 과제와 무관한 대화가 작업 기록으로 분류돼 라우팅 채점이 흐려진다.
"""

from __future__ import annotations

import datetime
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_codex_seeds", ROOT / "scripts" / "build_codex_seeds.py"
)
assert _spec and _spec.loader
seeds = importlib.util.module_from_spec(_spec)
sys.modules["build_codex_seeds"] = seeds
_spec.loader.exec_module(seeds)


def stamp(date: str) -> int:
    """그 날 정오. 날짜 경계에서 흔들리지 않게 한낮을 쓴다."""
    day = datetime.datetime.strptime(date, "%Y-%m-%d").replace(hour=12)
    return int(day.timestamp() * 1000)


def message(role: str, text: str, date: str, phase: str | None = None) -> dict:
    return {
        "role": role,
        "phase": phase,
        "text": text,
        "created_at_ms": stamp(date),
    }


class MessageFilterTests(unittest.TestCase):
    def test_commentary_is_dropped(self) -> None:
        """진행 안내는 근거가 아니다. 내보낸 답변의 2/3가 여기 해당한다."""
        kept = message("assistant", "결론은 이렇습니다", "2026-07-01", "final_answer")
        dropped = message("assistant", "파일을 읽고 있습니다", "2026-07-01", "commentary")
        self.assertTrue(seeds.usable(kept))
        self.assertFalse(seeds.usable(dropped))

    def test_user_messages_are_always_kept(self) -> None:
        self.assertTrue(seeds.usable(message("user", "이거 확인해줘", "2026-07-01")))


class MaskTests(unittest.TestCase):
    def test_keys_and_paths_are_masked(self) -> None:
        raw = (
            "키는 sk-or-v1-" + "a1b2c3d4" * 8 + " 이고 "
            "메일은 someone@example.com, 경로는 C:\\Users\\Tester\\Desktop 이다"
        )
        out = seeds.mask(raw)
        self.assertIn("sk-or-v1-REDACTED", out)
        self.assertNotIn("a1b2c3d4a1b2c3d4", out)
        self.assertIn("<메일주소>", out)
        self.assertNotIn("someone@example.com", out)
        self.assertIn("~\\Desktop", out)
        self.assertNotIn("Tester", out)

    def test_real_names_are_not_masked(self) -> None:
        """시드 폴더는 gitignore 대상이고 기존 시드도 실명을 쓴다.

        여기서 이름을 지우면 "누가 무엇을 하라고 했나" 질문이 성립하지
        않는다. 공개 저장소에 올리지 않는 것으로 막는다.
        """
        self.assertEqual(seeds.mask("홍길동이 검토했다"), "홍길동이 검토했다")


class DocumentTests(unittest.TestCase):
    def test_header_marks_tier_and_kind(self) -> None:
        body = "## 요청\n\n이거 봐줘\n\n## 답변\n\n봤습니다"
        text = seeds.document(
            date="2026-07-01",
            title="2026-07-01 Codex 작업기록 시험",
            body=body,
            origin="시험",
            relevant=True,
        )
        self.assertIn("memory_tier: mtm", text)
        self.assertIn("source_type: codex_worklog", text)
        self.assertIn("project: 진행 과제", text)
        self.assertTrue(text.startswith("---\n"))

    def test_offtopic_thread_is_marked_as_noise(self) -> None:
        """버리지 않고 방해 문서로 둔다. 지금 평가셋에 없는 축이다."""
        text = seeds.document(
            date="2026-07-01",
            title="제목",
            body="## 요청\n\n게임 파티 짜줘",
            origin="시험",
            relevant=False,
        )
        self.assertIn("source_type: codex_offtopic", text)
        self.assertIn("project: 기타", text)

    def test_summary_drops_section_headings(self) -> None:
        body = "## 요청\n\n라우팅 재측정 결과\n\n## 답변\n\n73/80이다"
        summary = seeds.summary_of(body)
        self.assertNotIn("##", summary)
        self.assertIn("라우팅 재측정 결과", summary)


class SlugTests(unittest.TestCase):
    def test_first_line_only(self) -> None:
        """제목에 첫 메시지를 그대로 쓴 스레드가 있다(머리글·경로·코드)."""
        piece = seeds.slug("# Files mentioned by the user:\n\n## 붙임1. 어쩌고")
        self.assertNotIn("\n", piece)
        self.assertTrue(piece.startswith("Files"))

    def test_korean_survives(self) -> None:
        self.assertEqual(seeds.slug("JEV 조사 및 적용 지점 확인"), "JEV-조사-및-적용-지점-확인")

    def test_empty_title_falls_back(self) -> None:
        self.assertEqual(seeds.slug("   "), "무제")


class BuildTests(unittest.TestCase):
    """내보내기 폴더를 만들어 끝에서 끝까지 돌린다."""

    def _export(self, base: Path, threads: list[dict]) -> str:
        import json

        folder = base / "내보내기"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "대화_데이터.json").write_text(
            json.dumps({"threads": threads}, ensure_ascii=False), encoding="utf-8"
        )
        return folder.name

    def test_thread_splits_by_day(self) -> None:
        import tempfile

        long_text = "라우팅 세미나 에이전트 메모리 프리페치 " * 30
        thread = {
            "title": "이틀에 걸친 작업",
            "messages": [
                message("user", "첫날 " + long_text, "2026-07-01"),
                message("assistant", long_text, "2026-07-01", "final_answer"),
                message("user", "이튿날 " + long_text, "2026-07-02"),
                message("assistant", long_text, "2026-07-02", "final_answer"),
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            name = self._export(base, [thread])
            seeds.EXPORT_BASE = base
            out = base / "out"
            seeds.build([name], out, write=True)
            written = sorted(p.name for p in (out / "mtm").iterdir())
        self.assertEqual(len(written), 2, written)
        self.assertTrue(written[0].startswith("2026-07-01-codex-"))
        self.assertTrue(written[1].startswith("2026-07-02-codex-"))

    def test_personal_payroll_thread_is_excluded(self) -> None:
        import tempfile

        payroll = "인건비 지급계획과 급여 명세서 정리 " * 20
        thread = {"title": "인건비", "messages": [message("user", payroll, "2026-07-01")]}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            name = self._export(base, [thread])
            seeds.EXPORT_BASE = base
            out = base / "out"
            seeds.build([name], out, write=True)
            self.assertFalse((out / "mtm").exists() and any((out / "mtm").iterdir()))

    def test_thin_day_is_skipped(self) -> None:
        import tempfile

        thread = {"title": "짧은 것", "messages": [message("user", "고마워", "2026-07-01")]}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            name = self._export(base, [thread])
            seeds.EXPORT_BASE = base
            out = base / "out"
            seeds.build([name], out, write=True)
            self.assertFalse((out / "mtm").exists() and any((out / "mtm").iterdir()))

    def test_internal_notes_are_skipped(self) -> None:
        import tempfile

        body = "라우팅 세미나 에이전트 메모리 " * 40
        threads = [
            {"title": "[내부 보조기록] Guardian review", "messages": [message("user", body, "2026-07-01")]},
            {"title": "실제 작업", "messages": [message("user", body, "2026-07-01")]},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            name = self._export(base, threads)
            seeds.EXPORT_BASE = base
            out = base / "out"
            seeds.build([name], out, write=True)
            written = sorted(p.name for p in (out / "mtm").iterdir())
        self.assertEqual(len(written), 1, written)
        self.assertIn("실제-작업", written[0])


if __name__ == "__main__":
    unittest.main()
