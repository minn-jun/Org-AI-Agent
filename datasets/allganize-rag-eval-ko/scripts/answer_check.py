"""Allganize 질문으로 에이전트를 실제 호출해 답변과 왕복 비용을 함께 기록한다.

검색 지표는 `scripts/bench_allganize.py`가 잰다. 이 스크립트는 그 다음 단계,
**답변이 맞는지**와 **한 답변에 왕복·토큰이 얼마나 드는지**를 한 실행에서 모은다.

    python datasets/allganize-rag-eval-ko/scripts/answer_check.py --dry      # 선정만 보기
    python datasets/allganize-rag-eval-ko/scripts/answer_check.py --limit 2  # 배선 확인용 2문항
    python datasets/allganize-rag-eval-ko/scripts/answer_check.py            # 20문항

- 정답은 데이터셋 원본의 `target_answer`다(제3자 작성). 판정은 **사람이 한다**.
  데이터셋이 제공하는 LLM 투표 판정은 사람 대비 오류가 약 8%라 쓰지 않는다.
- LTM에는 Allganize 청크만 올리고 STM·MTM은 비운다(합성 자료가 답에 섞이지 않게).
- 질문마다 새 세션으로 돌리고 **결과를 질문마다 저장한다.** 중간에 끊겨도 이미 지불한
  호출은 남는다. 이어서 돌리면 이미 끝난 질문은 건너뛴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
DATASET = HERE.parents[1]
ROOT = HERE.parents[3]                      # org_agent_mvp/
LABELS = DATASET / "labels" / "questions.jsonl"
CHUNKS = DATASET / "ltm" / "chunks.jsonl"
OUT_DIR = ROOT / "logs" / "answer_check"

#: 데이터셋의 근거 유형 비율(사용 가능한 299문항 기준).
#: 문항 수를 바꿀 때 이 비율을 유지한다 — 유형별 성공률이 크게 다르기 때문이다
#: (20문항 측정에서 문단형 6/10, 이미지형 0/4였다).
TYPE_RATIO = {"paragraph": 147, "text": 45, "table": 50, "image": 57}
#: 기본 20문항. `--count`로 바꾼다.
DEFAULT_COUNT = 20
QUOTA = {"paragraph": 10, "text": 3, "table": 3, "image": 4}
SEED_ORDER = ["finance", "public", "medical", "law", "commerce"]
#: 서로 다른 문서 수. 한 파일 1문항을 지키면 이 값이 문항 수의 상한이다.
DISTINCT_FILES = 58


def build_quota(count: int) -> dict[str, int]:
    """유형 비율을 지키면서 `count`문항으로 정원을 나눈다.

    최대 잔여법(Hare quota)을 쓴다. 단순 반올림은 합이 count와 어긋난다.
    각 유형에 최소 1건은 준다 — 이미지형처럼 적은 유형이 0건이 되면
    유형별 비교가 불가능해진다.
    """
    if count == DEFAULT_COUNT:
        return dict(QUOTA)
    total = sum(TYPE_RATIO.values())
    exact = {k: count * v / total for k, v in TYPE_RATIO.items()}
    quota = {k: max(1, int(v)) for k, v in exact.items()}
    while sum(quota.values()) < count:
        # 소수점이 가장 큰 유형부터 한 건씩 더한다.
        key = max(exact, key=lambda k: exact[k] - quota[k])
        quota[key] += 1
        exact[key] -= 1  # 같은 유형이 연속으로 받지 않도록 낮춘다
    while sum(quota.values()) > count:
        key = max(quota, key=lambda k: quota[k] - exact[k])
        if quota[key] <= 1:
            break
        quota[key] -= 1
    return quota


def load_questions() -> list[dict]:
    rows = []
    with LABELS.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not row.get("exclude"):
                rows.append(row)
    return rows


def select(rows: list[dict], quota: dict[str, int], max_per_file: int = 1) -> list[dict]:
    """유형별 정원 안에서 도메인을 돌아가며 뽑는다.

    qid가 붙어 있는 질문은 대체로 같은 문서라서, 도메인 안에서는 **간격을 두고** 고르고
    한 파일에서 `max_per_file`건을 넘지 않게 막는다. 순서가 고정이라 재현된다.
    """
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in sorted(rows, key=lambda r: int(r["qid"].split("_")[1])):
        by_type[row["context_type"]][row["domain"]].append(row)

    picked: list[dict] = []
    used_files: dict[str, int] = defaultdict(int)
    for ctype, need in quota.items():
        buckets = by_type.get(ctype, {})
        # 도메인마다 몇 건씩 가져올지 먼저 나눈다(고르게, 남으면 앞 도메인부터).
        doms = [d for d in SEED_ORDER if buckets.get(d)]
        if not doms:
            continue
        share = {d: need // len(doms) for d in doms}
        for d in doms[: need % len(doms)]:
            share[d] += 1
        for dom in doms:
            items = buckets[dom]
            take = share[dom]
            if take <= 0:
                continue
            step = max(1, len(items) // take)
            order = items[::step] + [it for it in items if it not in items[::step]]
            got = 0
            for row in order:
                if got >= take:
                    break
                name = Path(row["file_name"]).name
                if used_files[name] >= max_per_file:
                    continue
                picked.append(row)
                used_files[name] += 1
                got += 1
    return sorted(picked, key=lambda r: int(r["qid"].split("_")[1]))


def gold_document_ids(rows: list[dict]) -> dict[str, str]:
    """정답 파일명 → LTM document_id. 근거에 정답 문서가 들어왔는지 보는 데 쓴다."""
    mapping: dict[str, str] = {}
    with CHUNKS.open(encoding="utf-8") as fh:
        for line in fh:
            chunk = json.loads(line)
            meta = chunk.get("metadata") or {}
            path = meta.get("source_path") or ""
            doc_id = chunk.get("source_id") or ""
            if path and doc_id:
                mapping.setdefault(Path(path).name, doc_id)
    missing = [Path(r["file_name"]).name for r in rows
               if Path(r["file_name"]).name not in mapping]
    if missing:
        print(f"[warn] 청크에서 못 찾은 정답 파일 {len(set(missing))}건: {sorted(set(missing))[:3]}")
    return mapping


def summarize(result: dict, row: dict, gold_doc_id: str) -> dict:
    trace = result.get("trace", {})
    tokens = trace.get("tokens", {})
    tools = trace.get("tool_calls", []) or []
    prefetch = trace.get("prefetch", {})
    prefetch_sources = prefetch.get("sources", []) or []
    final_sources = trace.get("final_sources", []) or []
    return {
        "qid": row["qid"],
        "domain": row["domain"],
        "context_type": row["context_type"],
        "gold_page_has_text": row.get("gold_page_has_text"),
        "question": row["question"],
        "target_answer": row["target_answer"],
        "answer": result.get("answer", ""),
        "gold_file": row["file_name"],
        "gold_page_no": row["target_page_no"],
        "gold_doc_in_prefetch": gold_doc_id in prefetch_sources,
        "gold_doc_in_final": gold_doc_id in final_sources,
        "llm_calls": trace.get("llm_calls"),
        "tool_calls": len(tools),
        "retrieve_calls": sum(1 for t in tools if t.get("tool") == "retrieve_memory"),
        "expand_calls": sum(1 for t in tools if t.get("tool") == "expand_evidence"),
        "tool_queries": [
            {"tool": t.get("tool"), "tier": t.get("tier"), "query": t.get("query")}
            for t in tools
        ],
        "prompt_tokens": tokens.get("prompt_tokens"),
        "completion_tokens": tokens.get("completion_tokens"),
        "total_tokens": tokens.get("total_tokens"),
        "analyzer_tokens": (tokens.get("analyzer") or {}).get("total_tokens"),
        "agent_tokens": (tokens.get("agent") or {}).get("total_tokens"),
        "stopped_reason": trace.get("stopped_reason"),
        "analyzer_filters": (trace.get("query_analysis") or {}).get("filters"),
        "turn_log_path": str(result.get("turn_log_path") or ""),
        "verdict": "",          # 사람이 채운다: 정답 / 부분 / 오답 / 근거없음
        "verdict_note": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="앞에서 N문항만 실행")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"선정할 문항 수. 유형 비율은 유지한다 (기본 {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--max-per-file",
        type=int,
        default=1,
        help=f"한 파일에서 뽑는 문항 수. 1이면 문항 수 상한이 {DISTINCT_FILES}건이다",
    )
    parser.add_argument("--qids", nargs="*", help="특정 질문만 실행")
    parser.add_argument("--dry", action="store_true", help="선정 결과만 출력")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--retries", type=int, default=3, help="질문당 재시도 횟수")
    parser.add_argument("--rescore", action="store_true",
                        help="이미 저장된 결과의 근거 판정만 턴 로그로 다시 계산")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = load_questions()
    quota = build_quota(args.count)
    if args.count > DISTINCT_FILES and args.max_per_file <= 1:
        print(
            f"[warn] 서로 다른 문서가 {DISTINCT_FILES}개뿐이다. "
            f"--count {args.count}를 채우려면 --max-per-file 2가 필요하다"
        )
    picked = select(rows, quota, max_per_file=args.max_per_file)
    if args.qids:
        wanted = set(args.qids)
        picked = [r for r in rows if r["qid"] in wanted]
    if args.limit:
        picked = picked[: args.limit]

    print(f"선정 {len(picked)}문항  정원 {quota}  파일당 최대 {args.max_per_file}")
    for row in picked:
        print(f"  {row['qid']:>6}  {row['domain']:<8} {row['context_type']:<10} "
              f"{row['question'][:40]}")
    if args.dry:
        return 0

    if args.rescore:
        gold_map = gold_document_ids(rows)
        by_qid = {r["qid"]: r for r in rows}
        changed = 0
        for path in sorted(args.out.glob("q_*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            log_path = Path(record.get("turn_log_path") or "")
            if not log_path.exists():
                print(f"[skip] {record['qid']} 턴 로그 없음")
                continue
            log = json.loads(log_path.read_text(encoding="utf-8"))
            trace = log.get("trace", {})
            gold_doc_id = gold_map.get(Path(by_qid[record["qid"]]["file_name"]).name, "")
            record["gold_doc_id"] = gold_doc_id
            record["gold_doc_in_prefetch"] = gold_doc_id in (trace.get("prefetch", {}).get("sources") or [])
            record["gold_doc_in_final"] = gold_doc_id in (trace.get("final_sources") or [])
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            changed += 1
        print(f"재계산 {changed}건")
        return 0

    # LTM은 Allganize 청크만, STM·MTM은 비운다.
    os.environ["LTM_CORPUS"] = str(CHUNKS)
    os.environ.setdefault("MEMORY_ROOT", "datasets/allganize-rag-eval-ko/_empty_tiers")
    sys.path.insert(0, str(ROOT))

    from org_agent_mvp.__main__ import build_runtime   # CLI와 같은 조립 경로를 쓴다
    from org_agent_mvp.config import AppConfig

    args.out.mkdir(parents=True, exist_ok=True)
    turn_log_dir = args.out / "turns"
    turn_log_dir.mkdir(exist_ok=True)

    gold_map = gold_document_ids(rows)
    config = AppConfig.load()
    print(f"LTM 코퍼스: {config.ltm_corpus_path}")
    print(f"모델: analyzer={config.query_analyzer_model} / agent={config.agent_model}")
    runtime = build_runtime(config, use_mock=False)

    done = 0
    for row in picked:
        out_path = args.out / f"{row['qid']}.json"
        if out_path.exists():
            print(f"[skip] {row['qid']} (이미 있음)")
            continue
        gold_doc_id = gold_map.get(Path(row["file_name"]).name, "")
        for attempt in range(1, args.retries + 1):
            started = time.time()
            try:
                result = runtime.run(row["question"], log_dir=turn_log_dir)
            except Exception as exc:                       # 네트워크·API 오류
                print(f"[retry {attempt}/{args.retries}] {row['qid']}: {exc}")
                time.sleep(5 * attempt)
                continue
            record = summarize(result, row, gold_doc_id)
            record["elapsed_sec"] = round(time.time() - started, 1)
            out_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            done += 1
            print(
                f"[ok] {row['qid']} tokens={record['total_tokens']} "
                f"llm_calls={record['llm_calls']} retrieve={record['retrieve_calls']} "
                f"gold_doc={'Y' if record['gold_doc_in_final'] else 'N'} "
                f"{record['elapsed_sec']}s"
            )
            break
        else:
            print(f"[fail] {row['qid']} — 재시도 소진")

    print(f"\n완료 {done}문항. 결과: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
