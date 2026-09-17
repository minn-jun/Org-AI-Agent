"""parity_search.py 결과 전/후 비교. 검색 결과(순위·점수·접힘·유형·과제)와 문서 속성이 완전히 같은지 본다.

    python compare_parity.py            # before_stage{0..3}.json vs after_stage{0..3}.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ok = True
for stage in "0123":
    b = json.loads((HERE / f"before_stage{stage}.json").read_text(encoding="utf-8"))
    a = json.loads((HERE / f"after_stage{stage}.json").read_text(encoding="utf-8"))
    rows_diff = [(x["id"], x["variant"]) for x, y in zip(b["rows"], a["rows"]) if x != y]
    docs_diff = [k for k in b["docs"] if b["docs"][k][:3] != a["docs"].get(k, [None])[:3]]
    same = (len(b["rows"]) == len(a["rows"]) and not rows_diff and not docs_diff
            and b["stats"] == a["stats"] and b["source_types"] == a["source_types"])
    ok &= same
    print(f"stage {stage}: {'IDENTICAL' if same else 'DIFFERENT'} | 검색 {len(a['rows'])}건 중 다른 것 {len(rows_diff)} "
          f"| 문서 {len(a['docs'])}건 중 속성 다른 것 {len(docs_diff)}")
    for r in rows_diff[:5]:
        x = next(v for v in b["rows"] if (v["id"], v["variant"]) == r)
        y = next(v for v in a["rows"] if (v["id"], v["variant"]) == r)
        print("   ", r, "\n     before", x["results"][:3], "\n     after ", y["results"][:3])
    for k in docs_diff[:5]:
        print("    doc", k, b["docs"][k][:3], "->", a["docs"].get(k))
sys.exit(0 if ok else 1)
