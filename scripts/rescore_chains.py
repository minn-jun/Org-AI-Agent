"""저장된 연쇄 측정 기록을 **다시 채점한다.** 유료 호출 없음.

`answer_check_chains.py`가 턴마다 답변 전문 · 근거 글자 · 대화 기록 글자를
그대로 남긴다. 그래서 채점 규칙이 바뀌면 다시 부르지 않고 되채점할 수 있다.
판정 오탐을 고칠 때마다 다시 지불하지 않기 위한 것이다.

실행
----
    PYTHONIOENCODING=utf-8 python scripts/rescore_chains.py logs/answer_check_chains
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from answer_check_prentice import compact, loose  # noqa: E402

#: context_builder가 이전 턴에서 잘라 넘기는 길이. 대화 칸을 복원할 때 쓴다.
QUESTION_WINDOW = 300
SUMMARY_WINDOW = 500


def conversation_of(prior: list[dict]) -> str:
    """이전 턴 기록으로 `recent_session_turns` 칸을 복원한다.

    context_builder가 넘기는 것은 질문 300자 · 답변 500자 · source_ids다.
    기록에 그 셋이 다 있으므로 결정적으로 되만들 수 있다.
    """
    parts: list[str] = []
    for row in prior:
        parts.append(str(row["question"])[:QUESTION_WINDOW])
        parts.append(str(row["answer"])[:SUMMARY_WINDOW])
        parts.extend(str(source) for source in row.get("injected_sources", []))
    return " ".join(parts)


def rescore(row: dict, prior: list[dict]) -> dict:
    """답변 · 기대값 · 앵커를 느슨한 비교로 다시 판정한다.

    근거 글자는 기록에 없으므로 `delivered_expected`와
    `anchor_in_evidence`는 측정 당시 값을 그대로 쓴다.
    """
    row = dict(row)
    answer = loose(row["answer"])
    row["answer_has_expected_old"] = row["answer_has_expected"]
    row["answer_has_expected"] = any(loose(n) in answer for n in row["expect"])

    carry = dict(row.get("carry") or {})
    if carry:
        anchor = loose(carry["anchor"])
        prior_answer = str(prior[-1]["answer"]) if prior else ""
        position = loose(prior_answer).find(anchor)
        conversation = loose(row.get("conversation") or conversation_of(prior))
        carry["anchor_in_prior_answer_old"] = carry["anchor_in_prior_answer"]
        carry["anchor_in_prior_answer"] = position >= 0
        carry["anchor_cut_by_500"] = (
            position >= 0 and anchor not in loose(prior_answer[:SUMMARY_WINDOW])
        )
        carry["anchor_in_conversation_old"] = carry["anchor_in_conversation"]
        carry["anchor_in_conversation"] = anchor in conversation
        carry["conversation_only"] = (
            carry["anchor_in_conversation"] and not carry["anchor_in_evidence"]
        )
        row["carry"] = carry
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?",
                        default=ROOT / "logs" / "answer_check_chains")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    records: list[dict] = []
    for path in sorted(args.directory.glob("c*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        prior: list[dict] = []
        for row in payload["turns"]:
            records.append(rescore(row, prior))
            prior.append(row)
    if not records:
        print(f"{args.directory}에 기록이 없다")
        return 1

    total = len(records)
    flipped = [r for r in records
               if r["answer_has_expected"] != r["answer_has_expected_old"]]
    print(f"{args.directory.name}  턴 {total}개")
    print(f"  되채점으로 판정이 바뀐 턴 {len(flipped)}개")
    for row in flipped:
        before = "O" if row["answer_has_expected_old"] else "-"
        after = "O" if row["answer_has_expected"] else "-"
        print(f"    {row['qid']}  {before} → {after}   {row['question'][:40]}")
    print()
    print(f"  기대값이 근거에 전달   {sum(r['delivered_expected'] for r in records)}/{total}")
    print(f"  답변에 기대값         {sum(r['answer_has_expected'] for r in records)}/{total}")
    print(f"  근거 밖 숫자 있음      "
          f"{sum(1 for r in records if r['numbers_without_evidence'])}/{total}")
    print(f"  도구 호출 턴          {sum(1 for r in records if r['tool_calls'])}/{total}")

    carried = [r for r in records if r.get("carry")]
    if carried:
        keys = (
            ("anchor_in_prior_answer", "이전 답변에 값이 있었음"),
            ("anchor_cut_by_500", "500자 잘림으로 유실"),
            ("anchor_in_conversation", "대화 기록에 실림"),
            ("anchor_in_evidence", "근거에도 있음 (재검색)"),
            ("conversation_only", "대화에만 있음 (귀속 가능)"),
        )
        print(f"\n이어받기 검사 (앵커가 있는 턴 {len(carried)}개)")
        for key, label in keys:
            print(f"  {label:<24} {sum(r['carry'][key] for r in carried)}/{len(carried)}")
        only = [r for r in carried if r["carry"]["conversation_only"]]
        if only:
            print(f"    그중 답이 맞은 턴        "
                  f"{sum(r['answer_has_expected'] for r in only)}/{len(only)}")
            for row in only:
                print(f"      {row['qid']}  앵커 {row['carry']['anchor']!r}"
                      f"  정답 {'O' if row['answer_has_expected'] else '-'}")

    missed = [r for r in records if not r["answer_has_expected"]]
    if missed:
        print(f"\n기대값이 답변에 없는 턴 {len(missed)}개")
        for row in missed:
            print(f"  {row['qid']}  {'전달O' if row['delivered_expected'] else '전달-'}"
                  f"  {row['question'][:44]}")
            print(f"        기대 {row['expect']}")
            print(f"        답변 {compact(row['answer'])[:160]}")

    by_tier: dict[str, int] = defaultdict(int)
    for row in records:
        for tier, count in (row["prompt"].get("cards_by_tier") or {}).items():
            by_tier[tier] += count
    print(f"\n계층별 카드 {dict(by_tier)}  "
          f"참조 {sum(r['prompt'].get('reference_cards', 0) for r in records)}장")
    prompt_tokens = sum(
        int((row["tokens"] or {}).get("prompt_tokens", 0) or 0) for row in records
    )
    completion = sum(
        int((row["tokens"] or {}).get("completion_tokens", 0) or 0) for row in records
    )
    if prompt_tokens:
        print(f"입력 토큰 {prompt_tokens:,} (턴당 {prompt_tokens // total:,}) · "
              f"출력 {completion:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
