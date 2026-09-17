"""재정렬기 점수를 GPU로 미리 계산해 org_agent_mvp 재정렬 캐시에 넣는다.

CPU(기본 파이썬, torch CPU)에서 bge-reranker-v2-m3는 초당 1쌍 안팎이라 299문항 × 30쌍에 몇 시간이 걸린다.
검색 코드(org_agent_mvp/rerank.py)는 (모델, 질문, 청크) 해시로 캐시를 먼저 보므로,
같은 쌍의 점수를 GPU에서 계산해 캐시에 넣어 두면 검색 결과는 CPU로 계산한 것과 같은 순서가 된다.

    # 1) 후보 쌍 뽑기 — 기본 파이썬 (kiwipiepy 필요). 설정은 bench_allganize.py와 같게 준다
    python scripts/rerank_pairs.py dump --out ltm/rerank_pairs.jsonl

    # 2) GPU로 채점해 캐시에 넣기 — RAG venv (torch cu128, transformers)
    C:/ine_rag_venv/Scripts/python.exe scripts/rerank_pairs.py score --pairs ltm/rerank_pairs.jsonl --model BAAI/bge-reranker-v2-m3

점수는 sentence-transformers CrossEncoder와 같게 로짓에 시그모이드를 씌워 저장한다(순서는 로짓과 같다).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT.parent.parent
CACHE_ROOT = APP / "cache" / "rerank"

#: bench_allganize.py 최종 설정과 같아야 후보 30개가 같다.
FINAL_ENV = {
    "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_BM25_K1": "1.2",
    "RETRIEVER_TITLE_BONUS": "none", "QUERY_EXPANSIONS": "none", "RETRIEVER_DENSE": "0",
    "RETRIEVER_RERANK": "none", "LTM_DOC_META": "none",
}


def pair_key(query: str, passage: str) -> str:
    """org_agent_mvp/rerank.py `_pair_key`와 같아야 한다."""
    return hashlib.blake2b(f"{query}\x00{passage}".encode("utf-8"), digest_size=12).hexdigest()


def cache_path(model: str, max_length: int) -> Path:
    """org_agent_mvp/rerank.py `CrossEncoderScorer.cache_path`와 같아야 한다."""
    digest = hashlib.blake2b(f"{model}|{max_length}".encode("utf-8"), digest_size=8).hexdigest()
    return CACHE_ROOT / f"{digest}.jsonl"


def dump(args: argparse.Namespace) -> int:
    os.environ.update(FINAL_ENV)
    for item in args.set or []:
        key, _, value = item.partition("=")
        os.environ[key] = value
    sys.path.insert(0, str(APP))
    from org_agent_mvp.ltm_corpus import LtmCorpus
    from org_agent_mvp.memory_store import _expand_query
    from org_agent_mvp.tokenizer import tokenize

    corpus = LtmCorpus(ROOT / "ltm" / "chunks.jsonl")
    cases = []
    for name in ("dev", "test"):
        cases += [json.loads(l) for l in (ROOT / "ltm" / f"cases_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    seen = set()
    n = 0
    with open(ROOT / args.out, "w", encoding="utf-8") as fh:
        for c in cases:
            expanded = _expand_query(c["question"])
            tokens = tokenize(expanded)
            scores = corpus._score_chunks(tokens, expanded)
            query = expanded or " ".join(dict.fromkeys(tokens))
            for idx in sorted(scores, key=lambda i: -scores[i])[: args.top_n]:
                passage = f"{corpus.documents[corpus._chunk_doc[idx]].title}\n{corpus._chunk_text[idx]}"
                key = pair_key(query, passage)
                if key in seen:
                    continue
                seen.add(key)
                fh.write(json.dumps({"key": key, "query": query, "passage": passage}, ensure_ascii=False) + "\n")
                n += 1
    print(f"질문 {len(cases)} | 고유 쌍 {n} -> {ROOT / args.out}")
    return 0


def score(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pairs = [json.loads(l) for l in (ROOT / args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    target = cache_path(args.model, args.max_length)
    done = set()
    if target.exists():
        done = {json.loads(l)[0] for l in target.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [p for p in pairs if p["key"] not in done]
    print(f"쌍 {len(pairs)} | 이미 캐시 {len(pairs) - len(todo)} | 계산 {len(todo)} | 캐시 {target}", flush=True)
    if not todo:
        return 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=args.remote_code)
    model = (model.half() if device == "cuda" else model).to(device).eval()
    if args.remote_code:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from remote_code_fix import fix_model
        fix_model(model)
    # 긴 쌍끼리 묶으면 패딩이 줄어 빠르다
    todo.sort(key=lambda p: len(p["query"]) + len(p["passage"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with target.open("a", encoding="utf-8") as fh, torch.inference_mode():
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            enc = tok([p["query"] for p in batch], [p["passage"] for p in batch], padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits.float().view(-1).tolist()
            for p, logit in zip(batch, logits):
                fh.write(json.dumps([p["key"], 1.0 / (1.0 + math.exp(-logit))]) + "\n")
            if (start // args.batch) % 50 == 0:
                done_n = start + len(batch)
                print(f"  {done_n}/{len(todo)} ({done_n / (time.perf_counter() - t0):.1f}쌍/s)", flush=True)
    dt = time.perf_counter() - t0
    print(f"완료 {len(todo)}쌍 {dt:.0f}s ({len(todo) / dt:.1f}쌍/s, {device})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--out", default="ltm/rerank_pairs.jsonl")
    d.add_argument("--top-n", type=int, default=30)
    d.add_argument("--set", action="append")
    s = sub.add_parser("score")
    s.add_argument("--pairs", default="ltm/rerank_pairs.jsonl")
    s.add_argument("--model", required=True)
    s.add_argument("--remote-code", action="store_true", help="trust_remote_code 모델 (gte-multilingual 등, remote_code_fix 적용)")
    s.add_argument("--max-length", type=int, default=512)
    s.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    return {"dump": dump, "score": score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
