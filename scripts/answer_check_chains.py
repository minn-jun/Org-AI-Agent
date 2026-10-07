"""한 세션 여러 턴 — **실제 모델을 불러** 연쇄 질문에 답하게 하고 기록한다. 유료 호출.

`answer_check_prentice.py`와 다른 점은 하나다. 저쪽은 문항마다 새 세션을
열고, 이쪽은 **한 연쇄를 한 세션에서** 순서대로 묻는다. 그래서 이쪽만
답할 수 있는 질문이 있다 — "이전 턴의 답이 다음 턴에 쓸 수 있게 넘어오나".

이전 대화는 근거가 아니다
-------------------------
넘어오는 칸은 `recent_session_turns`이고, 들어가는 것은 이전 턴의
**질문 300자 · 답변 앞 500자 · source_ids**다. 근거 블록
(`prefetched_evidence`)과는 별개의 칸이다. 그래서 세 가지를 나눠 센다.

    anchor_in_prior_answer   이전 턴 답변에 그 값이 있었나      (생겼나)
    anchor_in_conversation   이번 턴 대화 기록에 실렸나          (넘어왔나)
    anchor_in_evidence       이번 턴 근거에도 있나              (다시 찾았나)

셋을 갈라야 "대화로 넘어와서 답했다"와 "다시 검색해서 답했다"를 구분할 수
있다. 둘 다면 대화 기록의 공을 주장할 수 없다. 그래서 가장 강한 증거는
`conversation_only` — 대화에는 있고 **근거에는 없는데** 답이 맞은 턴이다.

500자 잘림
----------
`answer_summary`는 답변의 앞 700자를 저장하고, 컨텍스트에 실을 때 다시
500자로 자른다. 답이 길면 뒤쪽 값은 **다음 턴에 존재하지 않는다.**
`anchor_cut_by_500`으로 이 경우를 따로 센다.

실행
----
    python scripts/answer_check_chains.py --dry
    SEED_EXCERPT_CHARS=6000 CONTEXT_MAX_EVIDENCE_CHARS=24000 \
        python scripts/answer_check_chains.py
    SESSION_EVIDENCE_LEDGER=1 ... --out logs/answer_check_chains_ledger
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CHAINS_PATH = ROOT / "datasets" / "prentice" / "session_chains.jsonl"
SEED = ROOT / "memory_seed_prentice"
OUT_DIR = ROOT / "logs" / "answer_check_chains"

from answer_check_prentice import (  # noqa: E402
    NUMBER_RE,
    compact,
    evidence_text,
    loose,
    prompt_shape,
)
from bench_session_chains import load_chains, runtime_context  # noqa: E402

#: context_builder가 answer_summary를 이만큼만 실어 보낸다.
SUMMARY_WINDOW = 500


def conversation_text(result: dict) -> str:
    """이번 턴 프롬프트에 실린 **대화 기록** 칸만 모은다. 근거는 빼고."""
    payload = runtime_context(result)
    return compact(
        json.dumps(payload.get("recent_session_turns", []), ensure_ascii=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cids", nargs="*", help="특정 연쇄만")
    parser.add_argument("--dry", action="store_true", help="선정만 출력")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.environ["MEMORY_ROOT"] = str(SEED)
    os.environ.pop("LTM_CORPUS", None)

    chains = load_chains(CHAINS_PATH)
    if args.cids:
        chains = {key: value for key, value in chains.items() if key in set(args.cids)}
    turn_total = sum(len(turns) for turns in chains.values())
    print(f"연쇄 {len(chains)}개 · 턴 {turn_total}개")
    for cid, turns in chains.items():
        print(f"  {cid}  {turns[0]['chain_title']}")
        for turn_row in turns:
            mark = "↳" if turn_row.get("needs_prior") else " "
            print(f"    {mark} {turn_row['t']}. {turn_row['question'][:52]:<54}"
                  f"기대 {turn_row['expect']}")
    if args.dry:
        return 0

    from org_agent_mvp.agent_runtime import AgentRuntime
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.openrouter_client import OpenRouterClient
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    config = AppConfig.load()
    if not config.api_key:
        sys.exit("OPENROUTER_API_KEY가 없다")
    store = build_memory_store(config)
    runtime = AgentRuntime(
        config=config,
        client=OpenRouterClient(config),
        memory_store=store,
        # 분석기는 규칙 기반으로 고정한다. 턴마다 계획이 흔들리면 연쇄에서
        # 차이의 원인을 가릴 수 없다.
        query_analyzer=RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary()),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n출력 {args.out}  발췌 {store.excerpt_chars}자  "
          f"근거 상한 {config.max_evidence_chars}자  "
          f"원장 {'ON' if config.session_evidence_ledger else 'OFF'}  "
          f"모델 {config.agent_model}\n")

    records: list[dict] = []
    for cid, turns in chains.items():
        path = args.out / f"{cid}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            records.extend(saved["turns"])
            print(f"{cid}  건너뜀 (이미 있음)")
            continue

        print(f"── {cid}  {turns[0]['chain_title']}")
        session_id: str | None = None
        prior_answer = ""
        chain_rows: list[dict] = []
        for turn_row in turns:
            result = None
            for attempt in range(1, args.retries + 1):
                try:
                    result = runtime.run(
                        turn_row["question"],
                        session_id=session_id,
                        log_dir=args.out / "turns",
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"   t{turn_row['t']}  실패 {attempt}/{args.retries}: {exc}")
                    if attempt < args.retries:
                        time.sleep(3)
            if result is None:
                break
            session_id = result["session_id"]

            answer = str(result.get("answer", ""))
            evidence = evidence_text(result)
            conversation = conversation_text(result)
            trace = result.get("trace", {})
            loose_answer = loose(answer)
            loose_evidence = loose(evidence)
            loose_conversation = loose(conversation)
            expect = turn_row["expect"]
            numbers = {n.strip(".,") for n in NUMBER_RE.findall(answer)}
            # 근거에도 대화 기록에도 출처 파일명에도 없는 숫자만 "근거 밖"으로
            # 센다. 이전 턴에서 넘어온 값을 할루시네이션으로 셀 수는 없고,
            # 모델이 출처로 적은 파일명의 숫자도 지어낸 값이 아니다
            # (`...GMT20260610-110130...`가 그렇게 걸렸다).
            haystack = loose(
                evidence + " " + conversation + " "
                + " ".join(trace.get("injected_sources", []))
                + " ".join(trace.get("reference_sources", []))
            )
            unsupported = sorted(n for n in numbers if loose(n) not in haystack)

            anchor = str(turn_row.get("prior_anchor") or "")
            anchor_report: dict = {}
            if anchor:
                # 앵커도 느슨하게 찾는다. 답변이 `Jev` · `2027년 9월 30일`로
                # 쓰면 원문 그대로는 못 만난다.
                position = loose(prior_answer).find(loose(anchor))
                window = loose(prior_answer[:SUMMARY_WINDOW])
                anchor_report = {
                    "anchor": anchor,
                    "prior_answer_chars": len(prior_answer),
                    "anchor_in_prior_answer": position >= 0,
                    # 이전 답에는 있었는데 500자 창 밖이라 실리지 못한 경우
                    "anchor_cut_by_500": position >= 0 and loose(anchor) not in window,
                    "anchor_in_conversation": loose(anchor) in loose_conversation,
                    "anchor_in_evidence": loose(anchor) in loose_evidence,
                }
                anchor_report["conversation_only"] = (
                    anchor_report["anchor_in_conversation"]
                    and not anchor_report["anchor_in_evidence"]
                )

            row = {
                "cid": cid,
                "qid": turn_row["qid"],
                "t": turn_row["t"],
                "question": turn_row["question"],
                "needs_prior": bool(turn_row.get("needs_prior")),
                "evidence_free": bool(turn_row.get("evidence_free")),
                "expect": expect,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt": prompt_shape(result),
                "conversation_chars": len(conversation),
                # 되채점할 수 있게 대화 기록 칸을 그대로 남긴다.
                "conversation": conversation,
                "delivered_expected": any(loose(n) in loose_evidence for n in expect),
                "answer": answer,
                "answer_has_expected": any(loose(n) in loose_answer for n in expect),
                "numbers_in_answer": sorted(numbers),
                "numbers_without_evidence": unsupported,
                "carry": anchor_report,
                "injected_sources": trace.get("injected_sources", []),
                "reference_sources": trace.get("reference_sources", []),
                "session_delivered_sources": trace.get("session_delivered_sources", []),
                "tool_calls": trace.get("tool_calls", []),
                "tokens": trace.get("tokens", {}),
                "stopped_reason": trace.get("stopped_reason", ""),
                "session_id": result.get("session_id", ""),
            }
            chain_rows.append(row)
            prior_answer = answer

            shape = row["prompt"]
            carry_mark = ""
            if anchor_report:
                carry_mark = (
                    f"  이어받음 {'O' if anchor_report['anchor_in_conversation'] else '-'}"
                    f"{'(대화만)' if anchor_report['conversation_only'] else ''}"
                )
            print(
                f"   t{turn_row['t']}  전달 {'O' if row['delivered_expected'] else '-'}"
                f"  정답 {'O' if row['answer_has_expected'] else '-'}"
                f"  근거밖숫자 {len(unsupported):>2}"
                f"  카드 {shape.get('card_count', 0)}({shape.get('cards_by_tier', {})})"
                f"  참조 {shape.get('reference_cards', 0)}"
                f"  대화 {row['conversation_chars']:>5,}자"
                f"  도구 {len(row['tool_calls'])}{carry_mark}"
            )

        if chain_rows:
            path.write_text(
                json.dumps(
                    {"cid": cid, "title": turns[0]["chain_title"], "turns": chain_rows},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            records.extend(chain_rows)
        print()

    if not records:
        return 0
    summarize(records)
    return 0


def summarize(records: list[dict]) -> None:
    total = len(records)
    followups = [row for row in records if row["t"] > 1]
    carried = [row for row in records if row.get("carry")]
    print("=" * 84)
    print(f"턴 {total}개 (후속 {len(followups)}개)")
    print(f"  기대값이 근거에 전달   {sum(row['delivered_expected'] for row in records)}/{total}")
    print(f"  답변에 기대값         {sum(row['answer_has_expected'] for row in records)}/{total}")
    print(f"  근거 밖 숫자 있음      "
          f"{sum(1 for row in records if row['numbers_without_evidence'])}/{total}")
    print(f"  도구 호출 턴          {sum(1 for row in records if row['tool_calls'])}/{total}")

    if carried:
        print(f"\n이어받기 검사 (앵커가 있는 턴 {len(carried)}개)")
        print(f"  이전 답변에 값이 있었음        "
              f"{sum(row['carry']['anchor_in_prior_answer'] for row in carried)}/{len(carried)}")
        print(f"  500자 잘림으로 유실            "
              f"{sum(row['carry']['anchor_cut_by_500'] for row in carried)}/{len(carried)}")
        print(f"  대화 기록에 실림               "
              f"{sum(row['carry']['anchor_in_conversation'] for row in carried)}/{len(carried)}")
        print(f"  근거에도 있음 (재검색)         "
              f"{sum(row['carry']['anchor_in_evidence'] for row in carried)}/{len(carried)}")
        only = [row for row in carried if row["carry"]["conversation_only"]]
        print(f"  대화에만 있음 (귀속 가능)      {len(only)}/{len(carried)}")
        if only:
            right = sum(row["answer_has_expected"] for row in only)
            print(f"    그중 답이 맞은 턴            {right}/{len(only)}")

    free = [row for row in records if row["evidence_free"]]
    if free:
        print(f"\n근거 없이 대화만으로 답하는 턴 {len(free)}개")
        for row in free:
            print(f"  {row['qid']}  정답 {'O' if row['answer_has_expected'] else '-'}"
                  f"  카드 {row['prompt'].get('card_count', 0)}"
                  f"  도구 {len(row['tool_calls'])}")

    missed = [row for row in records if not row["answer_has_expected"]]
    if missed:
        print(f"\n기대값이 답변에 없는 턴 {len(missed)}개")
        for row in missed:
            delivered = "전달O" if row["delivered_expected"] else "전달-"
            print(f"  {row['qid']}  {delivered}  {row['question'][:40]}")
            print(f"        기대 {row['expect']}")
            print(f"        답변 {compact(row['answer'])[:150]}")

    by_tier: dict[str, int] = defaultdict(int)
    for row in records:
        for tier, count in (row["prompt"].get("cards_by_tier") or {}).items():
            by_tier[tier] += count
    inputs = sum(int((row["tokens"] or {}).get("input", 0) or 0) for row in records)
    print(f"\n계층별 카드 {dict(by_tier)}")
    if inputs:
        print(f"입력 토큰 합 {inputs:,} · 턴당 평균 {inputs // total:,}")


if __name__ == "__main__":
    raise SystemExit(main())
