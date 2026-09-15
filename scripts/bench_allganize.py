"""Allganize RAG-Evaluation-Dataset-KO로 LTM 검색기만 잰다.

에이전트·질의 분석·계층 병합·LLM을 거치지 않는다. `LtmCorpus`를 직접 부른다.

    python scripts/bench_allganize.py run --split dev --label base
    python scripts/bench_allganize.py run --split all --label bm25 --set RETRIEVER_SCORER=bm25
    python scripts/bench_allganize.py compare logs/allganize/dev_base.json logs/allganize/dev_bm25.json
    python scripts/bench_allganize.py cv logs/allganize/all_A.json logs/allganize/all_B.json --out cv_A_B.json
    python scripts/bench_allganize.py table logs/allganize/dev_*.json

데이터 (저장소 밖): datasets/allganize-rag-eval-ko/ltm/  ← scripts/export_ltm_jsonl.py가 만든다
  chunks.jsonl, cases_dev.jsonl, cases_test.jsonl   (--split all = dev + test 299문항)

채점
  페이지 적중  search_chunks 상위 10개 중 "정답 파일이면서 page_nos에 정답 페이지가 있는" 첫 청크의 순위
  문서 적중    search(문서 단위) 상위 10개 중 정답 파일의 첫 순위
  Hit@k = 순위 <= k, MRR@10 = 1/순위 (없으면 0). 주 지표는 페이지 MRR@10.

compare: 두 실행의 질문별 페이지 RR 차이를 짝지어 부트스트랩(1,000회, 시드 20260915)으로
         95% 신뢰구간을 낸다. 구간이 0을 포함하면 동률로 본다.

cv: 5겹 교차검증으로 후보 선택 절차를 검증한다. 후보는 **단순한 것부터** 순서대로 준다.
    겹마다 나머지 4겹으로 고른다 — 가장 단순한 후보에서 시작해, 다음 후보가 현재 선택보다
    95% CI 하한 > 0으로 나을 때만 옮긴다. 고른 후보를 남은 1겹으로 채점한다.
    겹은 도메인 × 근거 유형 × 글자층 칸마다 섞어 돌아가며 나눈다(시드 20260915).

설정은 전부 환경변수로 준다. `--set`으로 준 것 외에는 아래 기본값(= 코드 기본값)을 명시해
실행 기록에 남긴다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "datasets" / "allganize-rag-eval-ko" / "ltm"
LOGS = ROOT / "logs" / "allganize"
TOP_K = 10
SEED = 20260915
BOOTSTRAP = 1000
FOLDS = 5

#: 기준선 설정 (코드 기본값). 과제명 가산점은 Allganize 문서에 project가 없어 자동으로 0이다.
BASE_ENV = {
    "RETRIEVER_TOKENIZER": "whitespace",
    "RETRIEVER_SCORER": "freq",
    "RETRIEVER_BM25_K1": "2.0",
    "RETRIEVER_TITLE_BONUS": "add",
    "QUERY_EXPANSIONS": "",          # 빈 값 = config/query_expansions.json
    "RETRIEVER_DENSE": "0",
    "RETRIEVER_RERANK": "none",
    "RETRIEVER_RERANK_TOP_N": "30",
    "LTM_DOC_META": "none",          # 범용 검색기: 문서 메타데이터 없이
}


def _git_commit() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


def _rank(results: list, hit) -> int | None:
    return next((i for i, item in enumerate(results, 1) if hit(item)), None)


def _metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in (1, 3, 5, 10)}
    out["mrr@10"] = sum(1 / r for r in ranks if r) / n
    return {k: round(v, 4) for k, v in out.items()}


def _load_cases(split: str) -> list[dict]:
    names = ["dev", "test"] if split == "all" else [split]
    cases = []
    for name in names:
        cases += [json.loads(l) for l in (DATA / f"cases_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return sorted(cases, key=lambda c: int(c["qid"].split("_")[1]))


def run(args: argparse.Namespace) -> int:
    env = dict(BASE_ENV)
    for item in args.set or []:
        key, _, value = item.partition("=")
        env[key] = value
    os.environ.update(env)
    sys.path.insert(0, str(ROOT))
    from org_agent_mvp.ltm_corpus import LtmCorpus
    from org_agent_mvp.memory_store import _expand_query
    from org_agent_mvp.tokenizer import tokenize

    cases = _load_cases(args.split)
    t0 = time.perf_counter()
    corpus = LtmCorpus(DATA / "chunks.jsonl")
    load_s = time.perf_counter() - t0

    rows = []
    q_times = []
    for c in cases:
        expanded = _expand_query(c["question"])
        tokens = tokenize(expanded)
        t = time.perf_counter()
        chunks = corpus.search_chunks(tokens, query_text=expanded, top_k=TOP_K)
        docs = corpus.search(tokens, query_text=expanded, top_k=TOP_K)
        q_times.append(time.perf_counter() - t)
        page_rank = _rank(chunks, lambda it: it[1]["document_id"] == c["doc_id"] and c["gold_page"] in it[1]["page_nos"])
        doc_rank = _rank(docs, lambda it: it[1]["source_ref"]["document_id"] == c["doc_id"])
        top = chunks[0][1] if chunks else None
        rows.append({
            "qid": c["qid"], "domain": c["domain"], "context_type": c["context_type"],
            "gold_page_has_text": c["gold_page_has_text"], "flag": c["flag"],
            "page_rank": page_rank, "doc_rank": doc_rank,
            "top1": {"title": top["title"], "page_nos": top["page_nos"]} if top else None,
        })

    def group(key):
        buckets = defaultdict(list)
        for r in rows:
            buckets[str(r[key])].append(r)
        return {k: {"n": len(v), "page": _metrics([r["page_rank"] for r in v])} for k, v in sorted(buckets.items())}

    summary = {
        "n": len(rows),
        "page": _metrics([r["page_rank"] for r in rows]),
        "doc": _metrics([r["doc_rank"] for r in rows]),
        "by_context_type": group("context_type"),
        "by_domain": group("domain"),
        "by_gold_page_has_text": group("gold_page_has_text"),
        "flagged": {r["qid"]: r["page_rank"] for r in rows if r["flag"]},
        "load_s": round(load_s, 2),
        "query_ms_mean": round(1000 * statistics.mean(q_times), 1) if q_times else 0,
    }
    record = {"label": args.label, "split": args.split, "env": env, "commit": _git_commit(),
              "corpus": corpus.stats() | {"source_types": corpus.source_types()},
              "summary": summary, "questions": rows}
    LOGS.mkdir(parents=True, exist_ok=True)
    out = LOGS / f"{args.split}_{args.label}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    p, d = summary["page"], summary["doc"]
    print(f"[{args.split}/{args.label}] n={len(rows)} 페이지 MRR@10 {p['mrr@10']:.3f} Hit@1 {p['hit@1']:.1%} "
          f"Hit@5 {p['hit@5']:.1%} Hit@10 {p['hit@10']:.1%} | 문서 MRR@10 {d['mrr@10']:.3f} Hit@1 {d['hit@1']:.1%} "
          f"| 로드 {summary['load_s']}s 질의 {summary['query_ms_mean']}ms -> {out}")
    return 0


def _rr(rank: int | None) -> float:
    return 1 / rank if rank else 0.0


def _bootstrap(diffs: list[float]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(BOOTSTRAP))
    return statistics.mean(diffs), means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    qa = {r["qid"]: r for r in a["questions"]}
    qb = {r["qid"]: r for r in b["questions"]}
    ids = sorted(set(qa) & set(qb), key=lambda q: int(q.split("_")[1]))
    diffs = [_rr(qb[q]["page_rank"]) - _rr(qa[q]["page_rank"]) for q in ids]
    mean, lo, hi = _bootstrap(diffs)
    better = sum(d > 0 for d in diffs)
    worse = sum(d < 0 for d in diffs)
    verdict = "동률 (95% CI가 0 포함)" if lo <= 0 <= hi else (f"{b['label']} 우세" if mean > 0 else f"{a['label']} 우세")
    result = {"a": a["label"], "b": b["label"], "n": len(ids), "page_mrr_a": a["summary"]["page"]["mrr@10"],
              "page_mrr_b": b["summary"]["page"]["mrr@10"], "diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
              "b_better": better, "b_worse": worse, "verdict": verdict}
    print(f"{a['label']} -> {b['label']}  n={len(ids)}  페이지 MRR@10 {result['page_mrr_a']:.3f} -> {result['page_mrr_b']:.3f}  "
          f"차이 {mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  좋아짐 {better} / 나빠짐 {worse}  => {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def _assign_folds(cases: list[dict]) -> dict[str, int]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for c in cases:
        cells[(c["domain"], c["context_type"], bool(c["gold_page_has_text"]))].append(c)
    rng = random.Random(SEED)
    fold_of: dict[str, int] = {}
    offset = 0
    for key in sorted(cells):
        items = sorted(cells[key], key=lambda c: int(c["qid"].split("_")[1]))
        rng.shuffle(items)
        for i, c in enumerate(items):
            fold_of[c["qid"]] = (offset + i) % FOLDS
        offset += len(items)
    return fold_of


def _select(runs: list[dict], ids: list[str]) -> tuple[int, list[dict]]:
    """단순한 후보부터 시작해 다음 후보가 CI 하한 > 0으로 나을 때만 옮긴다."""
    pick = 0
    steps = []
    for j in range(1, len(runs)):
        diffs = [_rr(runs[j]["q"][q]) - _rr(runs[pick]["q"][q]) for q in ids]
        mean, lo, hi = _bootstrap(diffs)
        moved = lo > 0
        steps.append({"from": runs[pick]["label"], "to": runs[j]["label"], "diff": round(mean, 4),
                      "ci95": [round(lo, 4), round(hi, 4)], "moved": moved})
        if moved:
            pick = j
    return pick, steps


def cv(args: argparse.Namespace) -> int:
    runs = []
    for path in args.files:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        runs.append({"label": r["label"], "q": {x["qid"]: x["page_rank"] for x in r["questions"]}})
    cases = _load_cases("all")
    ids_all = [c["qid"] for c in cases if all(c["qid"] in run_["q"] for run_ in runs)]
    fold_of = _assign_folds([c for c in cases if c["qid"] in set(ids_all)])

    folds = []
    held_rr: dict[str, float] = {}
    for k in range(FOLDS):
        train = [q for q in ids_all if fold_of[q] != k]
        test = [q for q in ids_all if fold_of[q] == k]
        pick, steps = _select(runs, train)
        for q in test:
            held_rr[q] = _rr(runs[pick]["q"][q])
        folds.append({"fold": k, "n_train": len(train), "n_test": len(test), "picked": runs[pick]["label"],
                      "held_out_mrr": round(statistics.mean(_rr(runs[pick]["q"][q]) for q in test), 4),
                      "steps": steps})
    full_pick, full_steps = _select(runs, ids_all)
    result = {
        "candidates": [r["label"] for r in runs],
        "n": len(ids_all),
        "overall_mrr": {r["label"]: round(statistics.mean(_rr(r["q"][q]) for q in ids_all), 4) for r in runs},
        "fold_sizes": dict(Counter(fold_of.values())),
        "folds": folds,
        "picks": dict(Counter(f["picked"] for f in folds)),
        "cv_held_out_mrr": round(statistics.mean(held_rr[q] for q in ids_all), 4),
        "full_data_pick": runs[full_pick]["label"],
        "full_data_steps": full_steps,
    }
    print("후보 (단순 -> 복잡):", " -> ".join(result["candidates"]), f"| n={len(ids_all)}")
    print("전체 299 페이지 MRR:", result["overall_mrr"])
    for f in folds:
        print(f"  겹 {f['fold']}: 선택 {f['picked']:28s} 남은 {f['n_test']}문항 MRR {f['held_out_mrr']:.3f} | "
              + " ; ".join(f"{s['to']} {s['diff']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]{'→옮김' if s['moved'] else ''}" for s in f["steps"]))
    print(f"선택 횟수 {result['picks']} | 교차검증(안 본 질문) 페이지 MRR {result['cv_held_out_mrr']:.3f} | 전체 데이터 규칙 선택 {result['full_data_pick']}")
    for s in full_steps:
        print(f"  전체: {s['from']} -> {s['to']} {s['diff']:+.4f} 95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] {'옮김' if s['moved'] else '유지'}")
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


def table(args: argparse.Namespace) -> int:
    print(f"{'label':30s}{'p MRR':>8s}{'p H@1':>8s}{'p H@5':>8s}{'p H@10':>8s}{'d MRR':>8s}{'d H@1':>8s}"
          f"{'표':>7s}{'이미지':>7s}{'문단':>7s}{'텍스트':>7s}{'글자층X':>8s}{'ms':>7s}")
    for path in args.files:
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        s = r["summary"]
        ct = s["by_context_type"]
        cell = lambda k: ct.get(k, {}).get("page", {}).get("mrr@10", float("nan"))  # noqa: E731
        no_text = s["by_gold_page_has_text"].get("False", {}).get("page", {}).get("mrr@10", float("nan"))
        print(f"{r['split'] + '/' + r['label']:30s}{s['page']['mrr@10']:8.3f}{s['page']['hit@1']:8.1%}{s['page']['hit@5']:8.1%}"
              f"{s['page']['hit@10']:8.1%}{s['doc']['mrr@10']:8.3f}{s['doc']['hit@1']:8.1%}"
              f"{cell('table'):7.3f}{cell('image'):7.3f}{cell('paragraph'):7.3f}{cell('text'):7.3f}{no_text:8.3f}"
              f"{s['query_ms_mean']:7.0f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test", "all"], required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--set", action="append", help="KEY=VALUE 환경변수 (여러 번)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out")
    v = sub.add_parser("cv")
    v.add_argument("files", nargs="+", help="--split all 실행 결과. 단순한 후보부터 순서대로")
    v.add_argument("--out")
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    args = ap.parse_args()
    return {"run": run, "compare": compare, "cv": cv, "table": table}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
