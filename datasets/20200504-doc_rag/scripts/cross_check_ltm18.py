"""20200504 LTM 단독 18건 교차 확인 — Allganize dev에서 고른 설정이 과제 폴더에서도 무너지지 않는가.

    python cross_check_ltm18.py [--dense]

네 가지를 잰다 (org_agent_mvp/scripts/bench_ltm_only.py의 run()을 그대로 쓴다).
  기준선 설정  × 문서 메타데이터 있음 / 없음
  최종 설정    × 문서 메타데이터 있음 / 없음

"메타데이터 없음" = 20200504 전용 전처리(enrich_doc_meta.py) 없이 범용 검색기만.
결과: org_agent_mvp/logs/allganize/cross_check_ltm18.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP = Path(r"C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp")
sys.path.insert(0, str(APP / "scripts"))
sys.path.insert(0, str(APP))

import bench_ltm_only  # noqa: E402

BASE = {
    "RETRIEVER_TOKENIZER": "whitespace", "RETRIEVER_SCORER": "freq", "RETRIEVER_BM25_K1": "2.0",
    "RETRIEVER_TITLE_BONUS": "add", "QUERY_EXPANSIONS": "", "RETRIEVER_DENSE": "0",
}
FINAL = {
    "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_BM25_K1": "1.2",
    "RETRIEVER_TITLE_BONUS": "none", "QUERY_EXPANSIONS": "none", "RETRIEVER_DENSE": "0",
}
#: 20200504 기존 2단계(형태소+BM25, k1 2.0, 제목 +2.0, 동의어 켬). 이전 측정(LTM MRR 0.451~0.467)과 잇기 위한 참고값.
PREV_STAGE2 = {
    "RETRIEVER_TOKENIZER": "morph", "RETRIEVER_SCORER": "bm25", "RETRIEVER_BM25_K1": "2.0",
    "RETRIEVER_TITLE_BONUS": "add", "QUERY_EXPANSIONS": "", "RETRIEVER_DENSE": "0",
}


def main() -> int:
    final = dict(FINAL)
    if "--dense" in sys.argv:
        final["RETRIEVER_DENSE"] = "1"
    configs = [("기준선", BASE), ("이전 2단계", PREV_STAGE2), ("최종", final)]
    metas = ("있음", "없음")
    out_name = "cross_check_ltm18.json"
    if "--rerank" in sys.argv:
        # 재정렬은 최종 설정에만 붙인다. 코퍼스(2.6만 청크)와 큰 모델을 한 프로세스에서 여러 번 올리면
        # 메모리 부족으로 죽어서(2026-09-15), 메타데이터 한 가지씩 따로 실행한다: --meta 있음|없음
        model = sys.argv[sys.argv.index("--rerank") + 1]
        meta_arg = sys.argv[sys.argv.index("--meta") + 1] if "--meta" in sys.argv else "있음"
        configs = [("최종+재정렬", {**final, "RETRIEVER_RERANK": model})]
        metas = (meta_arg,)
        out_name = f"cross_check_ltm18_rerank_meta_{'on' if meta_arg == '있음' else 'off'}.json"
        top_n = os.environ.get("RETRIEVER_RERANK_TOP_N", "").strip()
        if top_n and top_n != "30":
            # 후보 수를 바꾼 실행은 따로 저장한다 (기본 30개 결과를 덮어쓰지 않게)
            out_name = out_name.replace(".json", f"_top{top_n}.json")
            configs = [(f"최종+재정렬 상위{top_n}", configs[0][1])]
    rows = []
    for label, env in configs:
        env = {"RETRIEVER_RERANK": "none", **env}
        for meta in metas:
            run_env = {**env, "LTM_DOC_META": "" if meta == "있음" else "none"}
            r = bench_ltm_only.run(run_env, top_k=8)
            rows.append({"config": label, "doc_meta": meta, "env": run_env, "n": r["n"], "recall": round(r["recall"], 4),
                         "mrr": round(r["mrr"], 4), "hit1": round(r["hit1"], 4), "hit3": round(r["hit3"], 4),
                         "miss": r["miss"], "per_case": r["per_case"]})
            print(f"{label:10s} 메타데이터 {meta}  n={r['n']}  재현율 {r['recall']:.1%}  MRR {r['mrr']:.3f}  "
                  f"Hit@1 {r['hit1']:.1%}  Hit@3 {r['hit3']:.1%}  전멸 {r['miss']}", flush=True)
    out = APP / "logs" / "allganize" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
