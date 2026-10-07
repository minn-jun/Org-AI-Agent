"""한 세션 여러 턴 — 연쇄 질문의 **프롬프트 구성**을 센다. 유료 호출 없음.

왜 따로 재나
------------
`answer_check_prentice.py`는 문항마다 새 세션을 연다. 그래서 "이전 대화가
다음 턴에 실리는가"는 한 번도 측정된 적이 없다. 이 스크립트는 한 세션에
연쇄 질문을 순서대로 넣고, 턴마다 프롬프트를 되읽어 **이전 대화가 어느
칸으로 얼마나 들어오는지**를 센다.

이전 대화는 두 칸으로 들어온다
------------------------------
    recent_session_turns   이전 턴의 질문(300자) · 답변 앞부분(500자) ·
                           query_intent · source_ids · 도구 호출 3건.
                           **항상 켜져 있다**(SESSION_CACHE_TURNS=8).
    참조 카드               이미 본문을 준 문서를 본문 없이 한 줄로.
                           `SESSION_EVIDENCE_LEDGER=1`일 때만.

둘은 다른 것이다. 앞은 **대화 기록**이고 뒤는 **근거 목록**이다.
근거 블록(prefetched_evidence)에 이전 대화의 글이 섞여 들어가는 일은 없다.

읽는 법 — 이 스크립트가 재는 것과 못 재는 것
--------------------------------------------
LLM은 MockLLMClient다. 답변이 가짜이므로 `answer_summary`도 가짜다.
따라서 **이전 턴의 답에 있던 값이 다음 턴에 넘어가는지**는 여기서 못 센다
(그건 유료 `answer_check_chains.py`에서 센다). 여기서 확정하는 것은
LLM과 무관한 것뿐이다 —

    · 이전 턴 질문이 다음 턴 프롬프트에 실제로 들어갔나 (결정적)
    · 몇 턴까지 실리고, 글자 수가 얼마인가
    · 이전 턴의 source_ids가 함께 넘어갔나
    · 기대 계층의 카드가 후보에 있나
    · 원장을 켜면 재전송이 줄고 참조로 바뀌나

실행
----
    PYTHONIOENCODING=utf-8 python scripts/bench_session_chains.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHAINS_PATH = ROOT / "datasets" / "prentice" / "session_chains.jsonl"
SEED = ROOT / "memory_seed_prentice"


def load_chains(path: Path) -> dict[str, list[dict]]:
    chains: dict[str, list[dict]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            chains[row["cid"]].append(row)
    for turns in chains.values():
        turns.sort(key=lambda row: row["t"])
    return dict(chains)


def runtime_context(result: dict) -> dict:
    """프롬프트에 실린 [RUNTIME_CONTEXT] 블록을 되읽는다."""
    for message in result.get("messages", []):
        content = str(message.get("content", ""))
        if content.startswith("[RUNTIME_CONTEXT]"):
            body = content.split("[RUNTIME_CONTEXT]\n", 1)[1]
            return json.loads(body.split("\n[/RUNTIME_CONTEXT]", 1)[0])
    return {}


def measure(turn_row: dict, result: dict, prior: list[dict]) -> dict:
    payload = runtime_context(result)
    cards = payload.get("prefetched_evidence", [])
    recent = payload.get("recent_session_turns", [])
    trace = result.get("trace", {})

    # 카드는 tier를 "LTM"으로, 문항은 "ltm"으로 적는다. 소문자로 맞춘다.
    tiers: dict[str, int] = defaultdict(int)
    for card in cards:
        tiers[str(card.get("tier", "?")).lower()] += 1
    expect_tiers = [str(tier).lower() for tier in (turn_row.get("expect_tiers") or [])]

    # 이전 턴 질문이 글자로 실제 들어갔나. LLM과 무관하게 결정적이다.
    recent_blob = json.dumps(recent, ensure_ascii=False)
    carried = [row for row in prior if row["question"][:60] in recent_blob]
    prior_ids: set[str] = set()
    for item in recent:
        prior_ids |= {str(source) for source in item.get("source_ids", [])}

    return {
        "qid": turn_row["qid"],
        "t": turn_row["t"],
        "question": turn_row["question"],
        "needs_prior": bool(turn_row.get("needs_prior")),
        "evidence_free": bool(turn_row.get("evidence_free")),
        # ── 근거 쪽
        "candidates": trace.get("prefetch", {}).get("result_count", 0),
        "cards": len(cards),
        "cards_by_tier": dict(tiers),
        "bodies": sum(1 for card in cards if card.get("content_excerpt")),
        "references": len(trace.get("reference_sources", [])),
        "evidence_chars": len(json.dumps(cards, ensure_ascii=False)),
        "expect_tiers": expect_tiers,
        "expect_tier_hit": all(tiers.get(tier, 0) > 0 for tier in expect_tiers),
        # ── 이전 대화 쪽
        "recent_turns": len(recent),
        "recent_chars": len(recent_blob),
        "prior_questions_total": len(prior),
        "prior_questions_carried": len(carried),
        "prior_source_ids": len(prior_ids),
        # ── 그 외
        "tool_calls": len(trace.get("tool_calls", [])),
        "injected": list(trace.get("injected_sources", [])),
        "ledger_docs": len(trace.get("session_delivered_sources", [])),
    }


def run_chain(turns: list[dict], *, ledger: bool) -> list[dict]:
    from org_agent_mvp.agent_runtime import AgentRuntime
    from org_agent_mvp.config import AppConfig
    from org_agent_mvp.memory_store import build_memory_store
    from org_agent_mvp.mock_llm import MockLLMClient
    from org_agent_mvp.query_analyzer import RuleBasedQueryAnalyzer

    base = AppConfig.load()
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        config = replace(
            base, project_root=Path(temp_dir), session_evidence_ledger=ledger
        )
        store = build_memory_store(config)
        runtime = AgentRuntime(
            config=config,
            client=MockLLMClient(),
            memory_store=store,
            query_analyzer=RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary()),
        )
        session_id: str | None = None
        seen: set[str] = set()
        prior: list[dict] = []
        for turn_row in turns:
            result = runtime.run(turn_row["question"], session_id=session_id)
            session_id = result["session_id"]
            row = measure(turn_row, result, prior)
            injected = set(row["injected"])
            # 앞선 턴에 이미 본문을 준 문서를 또 본문으로 준 건수.
            row["resent"] = len(injected & seen)
            seen |= injected | set(result["trace"].get("tool_sources", []))
            prior.append(turn_row)
            rows.append(row)
    return rows


COLUMNS = (
    ("t", "턴", 3),
    ("candidates", "후보", 5),
    ("cards", "카드", 5),
    ("bodies", "본문", 5),
    ("references", "참조", 5),
    ("evidence_chars", "근거글자", 9),
    ("recent_turns", "이전턴", 7),
    ("recent_chars", "이전글자", 9),
    ("prior_source_ids", "이전출처", 9),
    ("resent", "재전송", 7),
    ("tool_calls", "도구", 5),
)


def print_chain(rows: list[dict], *, label: str) -> None:
    print(f"  ── {label}")
    print("    " + "".join(f"{name:>{width}}" for _, name, width in COLUMNS)
          + "  계층별 카드     기대  이전질문")
    for row in rows:
        line = "".join(f"{row[key]:>{width}}" for key, _, width in COLUMNS)
        tiers = ",".join(f"{key}{value}" for key, value in sorted(row["cards_by_tier"].items()))
        hit = "O" if row["expect_tier_hit"] else "-"
        carried = f"{row['prior_questions_carried']}/{row['prior_questions_total']}"
        print(f"    {line}  {tiers:<16} {hit}    {carried}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cids", nargs="*", help="특정 연쇄만")
    parser.add_argument("--out", type=Path, default=ROOT / "logs" / "session_chains.json")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.environ["MEMORY_ROOT"] = str(SEED)
    os.environ.pop("LTM_CORPUS", None)

    chains = load_chains(CHAINS_PATH)
    if args.cids:
        chains = {key: value for key, value in chains.items() if key in set(args.cids)}

    print(f"시드 {SEED.name}  연쇄 {len(chains)}개  "
          f"턴 {sum(len(turns) for turns in chains.values())}개\n")

    everything: dict[str, dict] = {}
    for cid, turns in chains.items():
        print("=" * 104)
        print(f"{cid}  {turns[0]['chain_title']}")
        for turn_row in turns:
            mark = "↳" if turn_row.get("needs_prior") else " "
            print(f"   {mark} {turn_row['t']}. {turn_row['question']}")
        off = run_chain(turns, ledger=False)
        on = run_chain(turns, ledger=True)
        print_chain(off, label="원장 OFF (기본값)")
        print_chain(on, label="원장 ON")
        everything[cid] = {"title": turns[0]["chain_title"], "off": off, "on": on}
        print()

    flat_off = [row for value in everything.values() for row in value["off"]]
    flat_on = [row for value in everything.values() for row in value["on"]]
    print("=" * 104)
    print("전체")
    followups = [row for row in flat_off if row["t"] > 1]
    carried = sum(row["prior_questions_carried"] for row in followups)
    total = sum(row["prior_questions_total"] for row in followups)
    print(f"  턴 수                         {len(flat_off)}")
    print(f"  2턴 이후 턴                   {len(followups)}")
    print(f"  이전 질문이 프롬프트에 실림    {carried}/{total}")
    print(f"  이전 출처 id가 함께 넘어온 턴  "
          f"{sum(1 for row in followups if row['prior_source_ids'])}/{len(followups)}")
    scored_off = [row for row in flat_off if row["expect_tiers"]]
    scored_on = [row for row in flat_on if row["expect_tiers"]]
    print(f"  기대 계층 카드 확보 (OFF)      "
          f"{sum(row['expect_tier_hit'] for row in scored_off)}/{len(scored_off)}")
    print(f"  기대 계층 카드 확보 (ON)       "
          f"{sum(row['expect_tier_hit'] for row in scored_on)}/{len(scored_on)}")
    for label, rows in (("OFF", flat_off), ("ON", flat_on)):
        print(f"  {label}  재전송 {sum(row['resent'] for row in rows):>3}건 · "
              f"본문 {sum(row['bodies'] for row in rows):>3}장 · "
              f"참조 {sum(row['references'] for row in rows):>3}장 · "
              f"근거 {sum(row['evidence_chars'] for row in rows):>7,}자 · "
              f"대화 {sum(row['recent_chars'] for row in rows):>6,}자")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(everything, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n기록 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
