"""진행 과제 시드로 **실제 모델을 불러** 답변을 받고 기록한다. 유료 호출.

무엇을 기록하나
---------------
질문마다 다음을 한 파일에 남긴다. 중간에 끊겨도 이미 지불한 호출은 남고,
이어서 돌리면 끝난 질문은 건너뛴다.

  프롬프트 구성   계층별 카드 수 · 근거 블록 글자 수 · 참조 카드 수
  전달            기대 답 문자열이 **전달된 근거** 안에 있었나
  답변            모델 답변 전문
  정답성          기대 답 문자열이 **답변** 안에 있나
  할루시네이션     답변에 나온 숫자가 전달된 근거에 **없는** 것이 있나
  왕복            도구 호출 수 · 토큰

할루시네이션 판정을 자동으로 하는 방법
--------------------------------------
답변에서 숫자(연도 · 금액 · 비율 · 식별번호)를 뽑아, 전달된 근거 글자 안에
그 숫자가 있는지 본다. 없으면 **근거 밖 숫자**로 센다. 사람이 읽어야 할
목록을 줄이는 1차 걸러내기이고, 판정 자체가 아니다 —
근거에 "2026년 4월"이 있고 답변이 "26년 4월"로 쓰면 걸린다.

실행
----
    python scripts/answer_check_prentice.py --dry
    SEED_EXCERPT_CHARS=6000 CONTEXT_MAX_EVIDENCE_CHARS=24000 \
        python scripts/answer_check_prentice.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = ROOT / "memory_seed_prentice"
OUT_DIR = ROOT / "logs" / "answer_check_prentice"

from check_prentice_delivery import ANSWER_CHECKS  # noqa: E402

#: 답변에서 뽑을 숫자. 1~2자리 단독 숫자는 문장 번호일 수 있어 3자 이상만 본다.
NUMBER_RE = re.compile(r"\d[\d,.\-]{2,}")


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", str(text))


def loose(text: str) -> str:
    """공백 · 콤마 · 숫자 구분자를 지운 비교용 문자열.

    2026-10-07에 1차 자동 판정이 3건을 오탐했다. 원인이 전부 표기 차이였다.

      PDF 추출 공백  과제번호 원문에 숫자 사이 공백이 들어간다
      구분자         원문 `2025-12-15` vs 답변 `2025.12.15`
      천 단위 콤마    기대값 `200000` vs 답변 `200,000`

    셋 다 같은 값인데 문자열로는 다르다. 비교 전에 지운다.
    """
    text = re.sub(r"\s+", "", str(text)).replace(",", "")
    return re.sub(r"[.\-/]", "", text)


def evidence_text(result: dict) -> str:
    """프롬프트에 실제로 들어간 근거 본문만 모은다."""
    blobs: list[str] = []
    for message in result.get("messages", []):
        content = str(message.get("content", ""))
        if content.startswith("[RUNTIME_CONTEXT]"):
            body = content.split("[RUNTIME_CONTEXT]\n", 1)[1]
            payload = json.loads(body.split("\n[/RUNTIME_CONTEXT]", 1)[0])
            for card in payload.get("prefetched_evidence", []):
                blobs.append(str(card.get("content_excerpt") or ""))
                for extra in card.get("additional_excerpts") or []:
                    blobs.append(str(extra.get("content") or ""))
        elif message.get("role") == "tool":
            blobs.append(content)
    return compact(" ".join(blobs))


def prompt_shape(result: dict) -> dict:
    for message in result.get("messages", []):
        content = str(message.get("content", ""))
        if content.startswith("[RUNTIME_CONTEXT]"):
            body = content.split("[RUNTIME_CONTEXT]\n", 1)[1]
            payload = json.loads(body.split("\n[/RUNTIME_CONTEXT]", 1)[0])
            cards = payload.get("prefetched_evidence", [])
            tiers: dict[str, int] = {}
            for card in cards:
                tier = str(card.get("tier", "?"))
                tiers[tier] = tiers.get(tier, 0) + 1
            return {
                "context_chars": len(content),
                "card_count": len(cards),
                "cards_by_tier": tiers,
                "reference_cards": sum(
                    1 for card in cards if card.get("delivered_in_turn") is not None
                ),
                "evidence_chars": len(
                    json.dumps(cards, ensure_ascii=False)
                ),
                "recent_turns": len(payload.get("recent_session_turns", [])),
            }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="앞에서 N문항만")
    parser.add_argument("--qids", nargs="*", help="특정 문항만")
    parser.add_argument("--dry", action="store_true", help="선정만 출력")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.environ["MEMORY_ROOT"] = str(SEED)
    os.environ.pop("LTM_CORPUS", None)

    checks = list(ANSWER_CHECKS)
    if args.qids:
        wanted = set(args.qids)
        checks = [c for c in checks if c[0] in wanted]
    if args.limit:
        checks = checks[: args.limit]

    print(f"문항 {len(checks)}건")
    for qid, question, needles in checks:
        print(f"  {qid:<6} {question[:44]:<46} 기대 {needles}")
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
        # 분석기는 규칙 기반으로 고정한다. LLM 분석기를 쓰면 실행마다 계획이
        # 흔들려 답변 차이의 원인이 섞인다.
        query_analyzer=RuleBasedQueryAnalyzer(vocabulary=store.filter_vocabulary()),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\n출력 {args.out}  발췌 {store.excerpt_chars}자  "
          f"근거 상한 {config.max_evidence_chars}자  모델 {config.agent_model}\n")

    summary = []
    for qid, question, needles in checks:
        path = args.out / f"{qid}.json"
        if path.exists():
            summary.append(json.loads(path.read_text(encoding="utf-8")))
            print(f"{qid}  건너뜀 (이미 있음)")
            continue
        for attempt in range(1, args.retries + 1):
            try:
                result = runtime.run(question, log_dir=args.out / "turns")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"{qid}  실패 {attempt}/{args.retries}: {exc}")
                if attempt == args.retries:
                    result = None
                time.sleep(3)
        if result is None:
            continue

        answer = str(result.get("answer", ""))
        evidence = evidence_text(result)
        trace = result.get("trace", {})
        loose_evidence = loose(evidence)
        loose_answer = loose(answer)
        numbers = {n.strip(".,") for n in NUMBER_RE.findall(answer)}
        unsupported = sorted(n for n in numbers if loose(n) not in loose_evidence)
        record = {
            "qid": qid,
            "question": question,
            "expected": needles,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "prompt": prompt_shape(result),
            "delivered_expected": any(loose(n) in loose_evidence for n in needles),
            "answer": answer,
            "answer_has_expected": any(loose(n) in loose_answer for n in needles),
            "numbers_in_answer": sorted(numbers),
            "numbers_without_evidence": unsupported,
            "injected_sources": trace.get("injected_sources", []),
            "tool_sources": trace.get("tool_sources", []),
            "cited_sources": trace.get("cited_sources", []),
            "tool_calls": trace.get("tool_calls", []),
            "tokens": trace.get("tokens", {}),
            "stopped_reason": trace.get("stopped_reason", ""),
            "session_id": result.get("session_id", ""),
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(record)
        shape = record["prompt"]
        print(
            f"{qid}  전달 {'O' if record['delivered_expected'] else '-'}"
            f"  정답 {'O' if record['answer_has_expected'] else '-'}"
            f"  근거밖숫자 {len(unsupported):>2}"
            f"  카드 {shape.get('card_count', 0)}({shape.get('cards_by_tier', {})})"
            f"  근거 {shape.get('evidence_chars', 0):,}자"
            f"  도구 {len(record['tool_calls'])}"
        )

    if not summary:
        return 0
    total = len(summary)
    print()
    print(f"전달 성공        {sum(r['delivered_expected'] for r in summary)}/{total}")
    print(f"답변에 기대값    {sum(r['answer_has_expected'] for r in summary)}/{total}")
    print(f"근거 밖 숫자 있음 {sum(1 for r in summary if r['numbers_without_evidence'])}/{total}")
    both = sum(
        1 for r in summary if r["delivered_expected"] and not r["answer_has_expected"]
    )
    print(f"전달됐는데 답 못 함 {both}/{total}  ← 모델 쪽 실패")
    missed = sum(
        1 for r in summary if not r["delivered_expected"] and r["answer_has_expected"]
    )
    print(f"전달 안 됐는데 답함 {missed}/{total}  ← 지식에서 꺼냈거나 추론. 확인 필요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
