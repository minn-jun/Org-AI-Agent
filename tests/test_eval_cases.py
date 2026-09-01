from __future__ import annotations

import json
import unittest
from pathlib import Path

from org_agent_mvp.memory_store import MemoryStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "eval_cases.jsonl"


class EvalCaseTests(unittest.TestCase):
    """평가셋이 실제 문서를 가리키는지 검증한다.

    gold 라벨에 오타가 있으면 평가 점수가 조용히 낮아지므로 여기서 막는다.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = [
            json.loads(line)
            for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        store = MemoryStore(PROJECT_ROOT / "memory_seed")
        cls.valid_ids = {f"ev_{doc.tier}_{doc.path.stem}" for doc in store.documents}

    def test_every_gold_id_points_to_a_real_document(self) -> None:
        for case in self.cases:
            for evidence_id in case["expected"]["gold_evidence_ids"]:
                with self.subTest(case=case["id"], evidence_id=evidence_id):
                    self.assertIn(evidence_id, self.valid_ids)

    def test_case_ids_are_unique(self) -> None:
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_routes_are_known_and_consistent_with_gold(self) -> None:
        for case in self.cases:
            expected = case["expected"]
            with self.subTest(case=case["id"]):
                self.assertIn(expected["route"], {"memory", "session", "direct"})
                if expected["route"] == "memory":
                    self.assertTrue(expected["gold_evidence_ids"])
                else:
                    self.assertEqual(expected["gold_evidence_ids"], [])


if __name__ == "__main__":
    unittest.main()
