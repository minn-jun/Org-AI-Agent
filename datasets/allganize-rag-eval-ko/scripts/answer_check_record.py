"""질문 · 예상 답변 · 나온 답변 · 판정 · tool call을 문항별 markdown으로 낸다.

    python datasets/allganize-rag-eval-ko/scripts/answer_check_record.py \
        logs/answer_check_delivery --also logs/answer_check --also logs/answer_check_strict \
        --labels 전달 기본 엄격 > 05-질문별-실행기록.md

첫 번째 폴더가 본문(전체 답변)이고, `--also`로 준 폴더는 판정만 나란히 놓는다.
답변을 셋 다 전문으로 실으면 읽을 수 없는 길이가 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(out_dir: Path) -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in out_dir.glob("q_*.json")
    }


def turn_log(record: dict) -> dict:
    path = record.get("turn_log_path")
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {}


def quote(text: str) -> str:
    """markdown 인용 블록. 빈 줄도 `>`로 이어야 블록이 끊기지 않는다."""
    lines = str(text).strip().splitlines() or [""]
    return "\n".join("> " + line if line.strip() else ">" for line in lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main_dir", type=Path)
    parser.add_argument("--also", action="append", default=[], type=Path)
    parser.add_argument("--labels", nargs="*", default=[])
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dirs = [args.main_dir, *args.also]
    labels = args.labels or [d.name for d in dirs]
    runs = [load(d) for d in dirs]
    main_run = runs[0]
    qids = sorted(main_run, key=lambda q: int(q.split("_")[1]))

    print("| qid | 유형 | " + " | ".join(f"{l} 판정" for l in labels) + " | 토큰 | 재검색 |")
    print("|---|---|" + "---|" * len(labels) + "---:|---:|")
    for qid in qids:
        r = main_run[qid]
        cells = [str(run.get(qid, {}).get("verdict") or "—") for run in runs]
        print(
            f"| [{qid}](#{qid}) | {r['context_type']} | " + " | ".join(cells)
            + f" | {r['total_tokens']:,} | {r['retrieve_calls']} |"
        )

    for qid in qids:
        r = main_run[qid]
        log = turn_log(r)
        trace = log.get("trace", {})
        print(f"\n---\n\n## {qid}\n")
        print(f"- 근거 유형: **{r['context_type']}** · 도메인: {r.get('domain', '')}")
        print(f"- 정답 문서: `{r.get('gold_file', '')}` {r.get('gold_page_no', '')}쪽"
              f" (페이지에 글이 있나: {'예' if r.get('gold_page_has_text') else '아니오'})")
        print(f"- 판정: **{r.get('verdict') or '미채점'}**", end="")
        others = [
            f"{label} {run.get(qid, {}).get('verdict') or '—'}"
            for label, run in zip(labels[1:], runs[1:])
        ]
        print(f" (다른 실행: {' / '.join(others)})" if others else "")
        if r.get("verdict_note"):
            print(f"- 대조 결과: {r['verdict_note']}")

        print(f"\n### 질문\n\n{r['question']}")
        print(f"\n### 예상 답변 (데이터셋의 `target_answer`)\n\n{quote(r['target_answer'])}")
        print(f"\n### 나온 답변\n\n{quote(r['answer'])}")

        print("\n### 실행 기록\n")
        print("| 항목 | 값 |")
        print("|---|---|")
        print(f"| 모델 호출 | {r['llm_calls']}회 |")
        print(f"| 토큰 | 입력 {r['prompt_tokens']:,} + 출력 {r['completion_tokens']:,} = **{r['total_tokens']:,}** |")
        print(f"| 질의 분석 필터 | `{json.dumps(r.get('analyzer_filters') or {}, ensure_ascii=False)}` |")
        print(f"| 검색이 찾은 근거 | {len(trace.get('retrieved_sources') or [])}건 |")
        print(f"| 프롬프트에 주입된 근거 | {len(trace.get('injected_sources') or [])}건 |")
        print(f"| 글자 수 제한으로 빠진 카드 | {len(trace.get('dropped_evidence_ids') or [])}건 |")
        print(f"| 답변에 나타난 근거 (추정) | {len(trace.get('cited_sources') or [])}건 |")
        print(f"| 종료 사유 | `{r.get('stopped_reason', '')}` |")

        executions = [
            t for t in log.get("tool_executions", [])
            if t.get("tool") in {"retrieve_memory", "expand_evidence"}
        ]
        if not executions:
            print("\n**tool call 없음.** 1차 컨텍스트의 근거만으로 답했다.")
            continue
        print(f"\n**tool call {len(executions)}회**\n")
        for item in executions:
            if item.get("tool") == "expand_evidence":
                print(f"{item['step']}. `expand_evidence` — 근거 {len(item.get('evidence_ids') or [])}건 요청, "
                      f"{item.get('expanded_count', 0)}건 반환")
            else:
                print(f"{item['step']}. `retrieve_memory` tier=`{item.get('tier', '')}` "
                      f"→ {item.get('result_count', 0)}건")
                print(f"   - 질의: `{item.get('query', '')}`")
            if item.get("reason"):
                print(f"   - 모델이 밝힌 이유: {item['reason']}")
            if item.get("status") not in {None, "completed"}:
                print(f"   - 상태: `{item['status']}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
