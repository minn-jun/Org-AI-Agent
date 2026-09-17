"""1회용: memory_store.py(STM/MTM)도 BM25+와 k1 설정을 따르게 한다. 기본값 동작은 그대로다."""
from pathlib import Path

p = Path(r"C:\ine_project\중기청_조직지식AI플랫폼\org_agent_mvp\org_agent_mvp\memory_store.py")
s = p.read_text(encoding="utf-8")
reps = [
    ("from .scoring import Bm25Params, freq_weight, scorer_name\n",
     "from .scoring import BM25_SCORERS, Bm25Params, bm25_params, freq_weight, scorer_name\n"),
    ('        use_bm25 = scorer_name("seed") == "bm25"\n',
     '        use_bm25 = scorer_name("seed") in BM25_SCORERS\n'
     '        params = bm25_params(self._bm25.n_docs, self._bm25.avg_len, scope="seed") if use_bm25 else None\n'),
    ("                    score += self._bm25.weight(\n",
     "                    score += params.weight(\n"),
]
for a, b in reps:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("memory_store.py bm25plus ok")
