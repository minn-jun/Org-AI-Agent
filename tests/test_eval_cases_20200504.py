from __future__ import annotations

import json
import unittest
from pathlib import Path

from org_agent_mvp.ltm_corpus import LtmCorpus, default_corpus_path
from org_agent_mvp.memory_store import MemoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "eval_cases_20200504.jsonl"
SEED_ROOT = PROJECT_ROOT / "memory_seed_20200504"
CORPUS_PATH = default_corpus_path(PROJECT_ROOT)


@unittest.skipUnless(CASES_PATH.exists(), "20200504 평가셋이 없다")
class Eval20200504CaseTests(unittest.TestCase):
    """실코퍼스 평가셋의 gold 라벨이 실제 문서를 가리키는지 검증한다.

    두 자료 모두 저장소에 없을 수 있어서, 없으면 해당 검사만 건너뛴다.

      STM/MTM  `memory_seed_20200504/` — gitignore 대상. 생성 스크립트로 만든다.
               `python scripts/build_lowvision_seed.py`
      LTM      `datasets/20200504-doc_rag/export/.../chunks.jsonl`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = [
            json.loads(line)
            for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.seed_ids: set[str] | None = None
        if SEED_ROOT.exists():
            store = MemoryStore(SEED_ROOT)
            cls.seed_ids = {f"ev_{doc.tier}_{doc.path.stem}" for doc in store.documents}
        cls.ltm_ids: set[str] | None = None
        if CORPUS_PATH.exists():
            corpus = LtmCorpus(CORPUS_PATH)
            cls.ltm_ids = {f"ev_ltm_{doc_id}" for doc_id in corpus.documents}

    def test_case_ids_are_unique(self) -> None:
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_seed_gold_ids_exist(self) -> None:
        if self.seed_ids is None:
            self.skipTest(
                f"시드가 없다: {SEED_ROOT}. "
                "python scripts/build_lowvision_seed.py 로 생성한다."
            )
        for case in self.cases:
            for evidence_id in case["expected"]["gold_evidence_ids"]:
                if evidence_id.startswith("ev_ltm_"):
                    continue
                with self.subTest(case=case["id"], evidence_id=evidence_id):
                    self.assertIn(evidence_id, self.seed_ids)

    def test_ltm_gold_ids_exist(self) -> None:
        if self.ltm_ids is None:
            self.skipTest(f"LTM 코퍼스가 없다: {CORPUS_PATH}")
        for case in self.cases:
            for evidence_id in case["expected"]["gold_evidence_ids"]:
                if not evidence_id.startswith("ev_ltm_"):
                    continue
                with self.subTest(case=case["id"], evidence_id=evidence_id):
                    self.assertIn(evidence_id, self.ltm_ids)

    def test_routes_are_known_and_consistent_with_gold(self) -> None:
        for case in self.cases:
            expected = case["expected"]
            with self.subTest(case=case["id"]):
                self.assertIn(expected["route"], {"memory", "session", "direct"})
                if expected["route"] == "memory":
                    self.assertTrue(expected["gold_evidence_ids"])
                else:
                    self.assertEqual(expected["gold_evidence_ids"], [])

    def test_project_filter_is_only_set_when_query_names_it(self) -> None:
        """`project`는 analyzer가 질의에서 뽑아야 할 값이다. 코퍼스 라벨이 아니다.

        질의에 과제명이 없는데 project를 채워 두면, 잡을 수 없는 값을 정답으로
        요구하게 되어 점수가 조용히 깎인다. 실제로 처음 만들 때 그렇게 해서
        프로젝트 필터 정확도가 3.6%로 나왔다.
        """
        for case in self.cases:
            project = case["expected"].get("project")
            with self.subTest(case=case["id"]):
                if project:
                    self.assertIn("과제", case["query"])


if __name__ == "__main__":
    unittest.main()
