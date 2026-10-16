"""질문 하나가 전 단계를 지나는 과정을 단계별로 펼쳐 본다.

왜 이 도구인가
--------------
지금까지 측정은 **끝 숫자**만 봤다 (tier@1 75/80, 정답 26/27). 중간에서 무엇이
들어가고 무엇이 빠졌는지는 로그를 직접 파야 알 수 있었다. 그래서 답이 틀렸을 때
"검색이 못 찾았나 · 찾았는데 안 실렸나 · 실렸는데 모델이 못 읽었나"를 가릴 수
없었다.

이 도구는 한 질문에 대해 여덟 단계를 순서대로 찍는다.

    ① 질의 분석      어느 저장소를 고르고 왜 골랐나 (Jev면 분기 확률까지)
    ② 저장소 검색     계층별로 몇 장을 봤나
    ③ 후처리         중복 제거 · 정규화 · 컷 · 계층 자리 · 상한에서 몇 장이 빠졌나
    ④ 근거 카드       최종 후보와 점수
    ⑤ 컨텍스트 주입   본문으로 실린 카드 · 참조만 준 카드 · 예산에서 탈락한 카드
    ⑥ LLM 왕복       호출마다 도구인가 최종 답변인가
    ⑦ 도구 실행       무엇을 검색했고 중복으로 몇 장이 빠졌나
    ⑧ 최종 답변       답과 기대값 대조

실행
----
    # 무료 (답변은 MockLLM, 분석기는 규칙)
    MEMORY_ROOT=memory_seed_prentice python scripts/trace_question.py \
        -q "연구노트는 몇 년 동안 보관해야 해?"

    # 분석기만 Jev (턴당 약 $0.00004)
    ... python scripts/trace_question.py --analyzer jev -q "..."

    # 답변까지 실제 모델 (턴당 약 $0.006)
    ... python scripts/trace_question.py --analyzer jev --real -q "..." --expect "5년"

    # 평가셋 문항으로
    ... python scripts/trace_question.py --qid r_01
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROUTING = ROOT / "datasets" / "prentice" / "routing_cases.jsonl"
CHAINS = ROOT / "datasets" / "prentice" / "session_chains.jsonl"
BAR = "─" * 96


def heading(step: str, title: str) -> None:
    print(f"\n{BAR}\n{step}  {title}\n{BAR}")


def load_case(qid: str) -> dict:
    """평가셋에서 문항을 찾는다. 라우팅과 연쇄 둘 다 본다."""
    for path in (ROUTING, CHAINS):
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("qid") == qid:
                return row
    raise SystemExit(f"{qid}를 평가셋에서 찾지 못했다")


def short(text: str, limit: int = 88) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--question", help="직접 물어볼 질문")
    parser.add_argument("--qid", help="평가셋 문항 id (예: r_01, c10_t1)")
    parser.add_argument("--expect", nargs="*", default=[], help="답에 있어야 할 문자열")
    parser.add_argument("--analyzer", choices=["rule", "llm", "jev"])
    parser.add_argument("--real", action="store_true", help="답변도 실제 모델로 (유료)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    for pair in args.set:
        key, _, value = pair.partition("=")
        os.environ[key.strip()] = value.strip()
    os.environ.setdefault("MEMORY_ROOT", "memory_seed_prentice")
    os.environ.pop("LTM_CORPUS", None)

    expect = list(args.expect)
    question = args.question
    expect_tiers: list[str] = []
    if args.qid:
        case = load_case(args.qid)
        question = case["question"]
        expect = expect or list(case.get("expect") or [])
        expect_tiers = [t.lower() for t in (case.get("expect_tiers") or [])]
    if not question:
        raise SystemExit("-q 또는 --qid 중 하나가 필요하다")

    from org_mem.agent_runtime import AgentRuntime
    from org_mem.builders import build_query_analyzer, describe
    from org_mem.config import AppConfig
    from org_mem.memory_store import build_memory_store
    from org_mem.mock_llm import MockLLMClient

    config = AppConfig.load()
    store = build_memory_store(config)
    analyzer = build_query_analyzer(config, store, kind=args.analyzer)
    if args.real:
        from org_mem.openrouter_client import OpenRouterClient

        client = OpenRouterClient(config)
    else:
        client = MockLLMClient()

    # 검색된 계층을 세려면 저장소 호출을 가로채야 한다. 점수 계산은 건드리지 않는다.
    # prefetch가 돈 뒤의 검색은 도구(retrieve_memory)가 부른 것이다. 나눠 센다.
    asked: list[tuple[str, int]] = []
    tool_asked: list[tuple[str, int]] = []
    phase = {"prefetch": True}
    original_retrieve = store.retrieve

    def counting_retrieve(*, tier, query, filters, top_k):
        result = original_retrieve(tier=tier, query=query, filters=filters, top_k=top_k)
        bucket = asked if phase["prefetch"] else tool_asked
        bucket.append((tier, int(result.get("result_count", 0))))
        return result

    store.retrieve = counting_retrieve  # type: ignore[method-assign]

    events: list[tuple[str, dict]] = []
    runtime = AgentRuntime(
        config=config, client=client, memory_store=store, query_analyzer=analyzer
    )

    print(f"질문   {question}")
    if expect:
        print(f"기대값 {expect}")
    if expect_tiers:
        print(f"기대 계층 {'+'.join(expect_tiers)}")
    print(f"\n설정   {json.dumps(describe(config, analyzer), ensure_ascii=False)}")

    def on_event(name: str, payload: dict) -> None:
        events.append((name, payload))
        if name == "prefetch_end":
            phase["prefetch"] = False

    result = runtime.run(question, event_callback=on_event)
    trace = result["trace"]
    by_event: dict[str, list[dict]] = {}
    for name, payload in events:
        by_event.setdefault(name, []).append(payload)

    # ─────────────────────────────────────────────── ① 질의 분석
    heading("①", "질의 분석")
    plan = trace["query_analysis"]
    print(f"분석기      {type(analyzer).__name__}")
    print(f"intent      {plan['intent']}")
    print(f"답변 출처    {plan['answer_source']}   검색 필요 {plan['memory_needed']}")
    print(f"계층 가중치   {plan['memory_weights']}")
    print(f"검색 문자열   {plan['query_rewrites']}")
    print(f"필터        {plan['filters'] or '없음'}")
    print(f"이유        {short(plan['reason'], 200)}")

    decision = getattr(analyzer, "last_decision", {}) or {}
    if decision.get("jev"):
        jev = decision["jev"]
        print(f"\nJev 분기     {jev['tier_branch']}  (confidence {jev['branch_confidence']:.2f})")
        ranked = sorted(jev["branch_probabilities"].items(), key=lambda kv: -kv[1])
        print("분기 확률    " + "  ".join(f"{n} {v:.2f}" for n, v in ranked if v > 0))
        print(f"비교 질문     {jev['is_comparison']:.2f}")
        print(f"검색할 계층   {'+'.join(decision['applied']['searched_tiers']) or '없음'}")
        if decision["overrides"]:
            for note in decision["overrides"]:
                print(f"  보정      {note}")
        print(f"지연·비용    {decision['latency_s']}초  ${decision.get('cost_usd') or 0:.7f}")

    # ─────────────────────────────────────────────── ② 저장소 검색
    heading("②", "저장소 검색")
    if not asked:
        print("검색하지 않았다 (검색 불필요 판정)")
    for tier, count in asked:
        print(f"  {tier.upper()}  후보 {count}장")
    skipped = [t for t in ("stm", "mtm", "ltm") if t not in {a for a, _ in asked}]
    if skipped:
        print(f"  건너뜀: {', '.join(t.upper() for t in skipped)}  (계층 가중치 0)")
    if tool_asked:
        print("  ── 아래는 도구(retrieve_memory)가 나중에 부른 검색이다")
        for tier, count in tool_asked:
            print(f"  {tier.upper()}  후보 {count}장")

    # ─────────────────────────────────────────────── ③ 후처리
    heading("③", "후처리 — 어디서 몇 장이 빠졌나")
    pf = (by_event.get("prefetch_end") or [{}])[0] or trace.get("prefetch", {})
    for key, label in (
        ("collected_count", "수집"),
        ("deduped_count", "중복 제거 후"),
        ("cut_threshold", "컷 기준점"),
        ("max_raw_score", "최고 원점수"),
    ):
        if key in pf:
            print(f"  {label:<14} {pf[key]}")
    print(f"  최종 후보       {len(trace.get('retrieved_sources', []))}개 문서")

    # ─────────────────────────────────────────────── ④ 근거 카드
    heading("④", "근거 카드")
    context = None
    for message in result["messages"]:
        body = str(message.get("content", ""))
        if body.startswith("[RUNTIME_CONTEXT]"):
            inner = body.split("[RUNTIME_CONTEXT]\n", 1)[1]
            context = json.loads(inner.split("\n[/RUNTIME_CONTEXT]", 1)[0])
            break
    cards = (context or {}).get("prefetched_evidence", [])
    print(f"{'계층':<6}{'본문':<6}{'점수':>8}  제목")
    for card in cards:
        has_body = "있음" if card.get("content_excerpt") else "참조"
        score = card.get("final_score", card.get("retrieval_score", ""))
        print(f"{str(card.get('tier','?')):<6}{has_body:<6}{score:>8}  {short(card.get('title',''), 58)}")
    got = [str(c.get("tier", "")).lower() for c in cards]
    if expect_tiers:
        ok1 = bool(got) and got[0] in expect_tiers
        print(f"\n기대 계층 {'+'.join(expect_tiers)}  1위 {got[0] if got else '-'}  "
              f"tier@1 {'O' if ok1 else 'X'}  전부 확보 {'O' if all(t in got for t in expect_tiers) else 'X'}")

    # ─────────────────────────────────────────────── ⑤ 컨텍스트 주입
    heading("⑤", "컨텍스트 주입")
    built = (by_event.get("context_built") or [{}])[0]
    for key, label in (
        ("candidate_count", "후보"),
        ("injected_count", "본문 주입"),
        ("dropped_count", "예산 탈락"),
        ("evidence_chars", "근거 글자"),
        ("max_evidence_chars", "근거 상한"),
        ("context_chars", "컨텍스트 전체"),
    ):
        if key in built:
            print(f"  {label:<14} {built[key]:,}" if isinstance(built[key], int) else f"  {label:<14} {built[key]}")
    if built.get("dropped_evidence_ids"):
        print(f"  탈락한 카드     {built['dropped_evidence_ids']}")
    recent = (context or {}).get("recent_session_turns", [])
    print(f"  이전 대화 칸    {len(recent)}턴  ← prefetched_evidence와 별개다")

    # ─────────────────────────────────────────────── ⑥ LLM 왕복
    heading("⑥", "LLM 왕복")
    for step in trace.get("reasoning_steps", []):
        print(f"  {step['llm_call']}회차  {step['decision']}" +
              (f"  근거 {step.get('source_count')}개" if step.get("source_count") is not None else ""))

    # ─────────────────────────────────────────────── ⑦ 도구 실행
    heading("⑦", "도구 실행")
    tool_calls = trace.get("tool_calls", [])
    if not tool_calls:
        print("  없음 — prefetch 근거로 바로 답했다")
    for call in tool_calls:
        print(f"  {json.dumps(call, ensure_ascii=False)[:200]}")

    # ─────────────────────────────────────────────── ⑧ 최종 답변
    heading("⑧", "최종 답변")
    answer = result.get("final_answer") or ""
    print(answer[:1200])
    if expect:
        from scripts.answer_check_prentice import loose  # noqa: E402

        blob = loose(answer)
        print()
        for value in expect:
            print(f"  기대값 {value!r}  {'O' if loose(value) in blob else 'X'}")

    tokens = trace.get("tokens", {})
    print(f"\n토큰  분석기 {tokens.get('analyzer', {}).get('prompt_tokens', 0):,} / "
          f"에이전트 {tokens.get('agent', {}).get('prompt_tokens', 0):,} 입력, "
          f"출력 {tokens.get('completion_tokens', 0):,}  "
          f"(추정값 {tokens.get('estimated')})")
    print(f"중단 이유  {trace.get('stopped_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
