"""재정렬기 CPU 속도 — 실제 후보 쌍(질문 하나당 상위 N개)으로 질문당 걸리는 시간을 잰다.

    python scripts/rerank_latency.py --model BAAI/bge-reranker-v2-m3 --top-n 30 --max-length 512
    python scripts/rerank_latency.py --model Alibaba-NLP/gte-multilingual-reranker-base --remote-code

ltm/rerank_pairs.jsonl은 질문별로 상위 30쌍이 순서대로 들어 있다(중복 쌍은 빠짐).
질문 앞쪽 --queries개를 골라, 질문마다 앞 --top-n쌍을 한 번에 채점한다. 첫 질문은 워밍업으로 뺀다.
기본 파이썬(torch CPU, 8스레드)으로 돌린다. 캐시는 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--queries", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--remote-code", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pairs = [json.loads(l) for l in (ROOT / "ltm" / "rerank_pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = [list(g)[: args.top_n] for _, g in groupby(pairs, key=lambda p: p["query"])]
    groups = [g for g in groups if len(g) >= args.top_n][: args.queries + 1]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=args.remote_code).eval()
    if args.remote_code:
        from remote_code_fix import fix_model
        fix_model(model)

    times = []
    with torch.inference_mode():
        for qi, group in enumerate(groups):
            t0 = time.perf_counter()
            for s in range(0, len(group), args.batch):
                b = group[s:s + args.batch]
                enc = tok([p["query"] for p in b], [p["passage"] for p in b], padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt")
                model(**enc)
            dt = time.perf_counter() - t0
            if qi:
                times.append(dt)
            print(f"  질문 {qi}{' (워밍업)' if not qi else ''}: {dt:.1f}s", flush=True)
    out = {"model": args.model, "top_n": args.top_n, "max_length": args.max_length, "batch": args.batch,
           "threads": torch.get_num_threads(), "queries": len(times),
           "sec_per_query": round(statistics.mean(times), 2), "pairs_per_sec": round(args.top_n / statistics.mean(times), 2)}
    print(json.dumps(out, ensure_ascii=False))
    log = ROOT / "logs" / "rerank_latency.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
