"""점수 함수. 단계별로 갈아 끼울 수 있게 한 곳에 모았다.

    RETRIEVER_SCORER=freq      빈도 기반 (기본, 0~1단계)
    RETRIEVER_SCORER=bm25      BM25 (2단계~)
    RETRIEVER_SCORER=bm25plus  BM25+ (BM25에 단어 등장 하한 δ를 더한다)
    RETRIEVER_BM25_K1=<수>     k1을 바꾼다 (기본 2.0)

## 왜 BM25인가

지금 쓰는 빈도 점수는 이렇다.

    score = Σ  1 + log(1 + f(t,d))
          t∈q

두 가지가 빠져 있다.

1. **흔한 단어와 드문 단어를 구분하지 않는다.**
   "사업"은 거의 모든 문서에 나오는데 "출연금"만큼 배점을 받는다.
2. **문서 길이를 보정하지 않는다.**
   긴 문서는 우연히도 질의어를 더 많이 포함한다.

BM25는 둘 다 넣는다.

    score = Σ  IDF(t) · ─────────f(t,d)·(k1+1)─────────
          t∈q            f(t,d) + k1·(1 − b + b·|d|/avgdl)

    IDF(t) = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))

- `IDF`가 흔한 단어의 배점을 깎는다.
- 분모의 `|d|/avgdl`이 긴 문서를 눌러 준다.
- `k1`은 빈도가 얼마나 빨리 포화되는지, `b`는 길이 보정을 얼마나 세게 할지다.

## BM25+

BM25는 길이 보정 때문에 긴 청크에서 단어가 한 번 나온 점수가 크게 줄어든다
(Allganize 청크 기준 tf=1일 때 가장 짧은 쪽 1.49, 가장 긴 쪽 0.53).
BM25+(Lv & Zhai, 2011)는 단어가 한 번이라도 나오면 하한 δ를 더한다.

    score = Σ  IDF(t) · [ ─────────f(t,d)·(k1+1)───────── + δ ]
          t∈q             f(t,d) + k1·(1 − b + b·|d|/avgdl)

δ는 논문 기본값 1.0으로 고정한다. IDF 식은 BM25와 같게 둬서 차이가 δ에서만 나오게 한다.

외부 라이브러리를 쓰지 않는다. 식이 짧고, 우리가 이미 역색인과 문서 길이를
들고 있어서 얹기만 하면 된다. 의존성이 늘면 재현 환경만 복잡해진다.
"""

from __future__ import annotations

import math
import os

#: 빈도 포화 계수. 통상 1.2~2.0을 쓴다.
#: 낮을수록 "한 번 나오든 열 번 나오든 비슷하다"에 가까워진다.
#:
#: 이 코퍼스에서는 관례적 기본값 1.2가 맞지 않았다. LTM 검색만 떼어 재보니
#:
#:     k1=1.2  MRR 0.392   재현율 50.5%
#:     k1=2.0  MRR 0.467   재현율 50.9%   <- 채택
#:
#: 과제 문서는 길고 같은 용어가 반복돼서, 빈도를 일찍 포화시키면
#: "진짜 그 주제인 문서"와 "한 번 스친 문서"가 구분되지 않는다.
#: (20200504 과제 LTM 18건 기준이다. 다른 코퍼스에서는 RETRIEVER_BM25_K1로 다시 잰다.)
K1 = 2.0

#: 길이 보정 계수. 0이면 길이를 무시하고, 1이면 완전히 보정한다.
#: 0.75가 관례적인 기본값이고, 이 코퍼스에서도 그대로 두는 게 나았다
#: (b=1.0이 MRR은 0.010 높지만 재현율이 2.8%p 낮다).
B = 0.75

#: BM25+ 하한. Lv & Zhai(2011)의 기본값이다.
BM25_PLUS_DELTA = 1.0

DEFAULT_SCORER = "freq"
BM25_SCORERS = frozenset({"bm25", "bm25plus"})
_SCORERS = frozenset({"freq"}) | BM25_SCORERS


def scorer_name(scope: str = "ltm") -> str:
    """점수 함수를 고른다. 계층별로 다르게 줄 수 있다.

    코퍼스 크기가 100배 차이나면 같은 점수 함수가 양쪽에 다 맞지 않는다.

        LTM        청크 26,031건. IDF가 의미 있고 길이도 제각각이라 BM25가 맞는다.
        STM/MTM    문서 85건. IDF가 거의 평평하고 길이도 고른다.
                   BM25를 씌우면 오히려 나빠졌다 — 실측으로 MRR 0.864 -> 0.680.

    RETRIEVER_SCORER      둘 다에 적용되는 기본값
    RETRIEVER_SCORER_SEED STM/MTM만 따로 지정 (없으면 위 값을 따른다)
    """
    base = os.environ.get("RETRIEVER_SCORER", DEFAULT_SCORER).strip().lower()
    if scope == "seed":
        base = os.environ.get("RETRIEVER_SCORER_SEED", base).strip().lower()
    return base if base in _SCORERS else DEFAULT_SCORER


def bm25_k1() -> float:
    """RETRIEVER_BM25_K1이 있으면 그 값, 없거나 숫자가 아니면 K1."""
    try:
        value = float(os.environ.get("RETRIEVER_BM25_K1", K1))
    except ValueError:
        return K1
    return value if value > 0 else K1


def freq_weight(tf: int) -> float:
    """0~1단계 점수. 문서 길이도 희소성도 보지 않는다."""
    return 1.0 + math.log1p(tf)


class Bm25Params:
    """코퍼스 단위로 한 번 계산해 두는 값.

    `avgdl`은 평균 문서 길이, `n_docs`는 문서 수다.
    IDF는 토큰마다 다르므로 조회할 때 계산한다(캐시해도 되지만
    질의어가 십여 개라 실측상 차이가 없었다).
    `delta`가 0이면 BM25, 양수면 BM25+다.
    """

    def __init__(self, n_docs: int, avg_len: float, k1: float = K1, b: float = B, delta: float = 0.0):
        self.n_docs = max(1, n_docs)
        self.avg_len = max(1.0, avg_len)
        self.k1 = k1
        self.b = b
        self.delta = delta

    def idf(self, df: int) -> float:
        """희소할수록 크다. df가 N에 가까우면 0에 수렴한다."""
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def weight(self, tf: int, doc_len: int, df: int) -> float:
        denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_len)
        if denom <= 0:
            return 0.0
        w = self.idf(df) * (tf * (self.k1 + 1.0)) / denom
        if self.delta and tf > 0:
            w += self.idf(df) * self.delta
        return w


def bm25_params(n_docs: int, avg_len: float, scope: str = "ltm") -> Bm25Params:
    """현재 설정(RETRIEVER_SCORER, RETRIEVER_BM25_K1)대로 BM25 계열 파라미터를 만든다."""
    delta = BM25_PLUS_DELTA if scorer_name(scope) == "bm25plus" else 0.0
    return Bm25Params(n_docs=n_docs, avg_len=avg_len, k1=bm25_k1(), delta=delta)


def describe() -> dict[str, object]:
    name = scorer_name()
    return {"scorer": name, "k1": bm25_k1(), "b": B,
            "delta": BM25_PLUS_DELTA if name == "bm25plus" else 0.0}


#: prefetch가 계층 점수를 합칠 때 무엇을 기준으로 1.0을 잡을지.
#:
#:   global  전역 최고점. 계층 간 점수가 비교 가능하다는 전제.
#:   tier    계층별 최고점. 각 계층 1등이 동률이 되고 tier_prior가 순위를 정한다.
#:   rrf     점수를 버리고 계층 안 순위만 쓴다. 점수 눈금이 달라도 섞인다.
#:   zscore  계층 안에서 몇 σ 튀는지로 맞춘다. 아래 설명.
#:   hybrid  global과 zscore의 기하평균. 둘 다 높아야 높아진다.
#:
#: BM25는 IDF의 N이 계층마다 달라 점수 눈금이 어긋난다. 그때 이 선택이 크게 갈린다.
#:
#: ## 눈금 격차가 계층 병합을 얼마나 망가뜨리나
#:
#: 실측했다. 60건 질의에서 계층별 최고점의 평균을 재고, 같은 설정의 계층 혼합 MRR과
#: 나란히 놓으면 단조 관계가 나온다.
#:
#:     설정              LTM/STM 점수비    계층 혼합 MRR
#:     freq 전부              1.50x          0.677
#:     bm25 전부              1.97x          0.429
#:     bm25 LTM만(2s)         2.44x          0.198
#:
#: 2s는 **LTM 단독 MRR이 0.461로 전 설정 중 최고**인데 전체는 최악이다.
#: 검색이 나빠진 게 아니라 병합에서 잃는다.
#:
#: ## tier / rrf가 대신이 되지 못하는 이유
#:
#: 둘 다 눈금은 맞추지만 "이 계층에 쓸 만한 게 있는가"를 같이 버린다.
#: 계층별 최고점으로 나누면 관련 없는 STM 문서도 자기 계층 1등이면 1.0을 받는다.
#: 그래서 LTM 단독 질문에서 STM/MTM이 자리를 뺏는다 — LTM MRR이 0.451에서
#: tier 0.176, rrf 0.110으로 무너졌다.
#:
#: ## zscore
#:
#: 계층 최고점이 아니라 **그 계층 후보 분포에서 얼마나 튀는지**로 맞춘다.
#:
#:     z = (점수 − 계층 평균) / 계층 표준편차
#:     normalized = 1 / (1 + exp(−z))
#:
#: 정답을 가진 계층은 top이 자기 풀 평균보다 크게 튀고, 볼 게 없는 계층은
#: 조금밖에 안 튄다. tier는 둘 다 1.0을 주지만 zscore는 갈라 놓는다.
#: 계층 안에서는 선형변환이라 순위가 그대로 보존된다.
#:
#: sigmoid를 씌우는 이유는 뒤쪽이 양수를 전제하기 때문이다 — tier_prior를
#: 곱하고, 상대 컷을 `최고점 × cut_ratio`로 잡는다. z는 음수가 될 수 있다.
DEFAULT_NORMALIZE = "global"

#: 후보가 이보다 적으면 평균·표준편차를 믿을 수 없다. 그 계층은 global로 처리한다.
ZSCORE_MIN_POOL = 3

#: hybrid에서 z 쪽에 주는 비중. 0이면 global과 같고, 1이면 zscore와 같다.
#:
#:     normalized = global^(1-w) · z^w
#:
#: 로그 공간의 선형 보간이라 w=0.5가 기하평균이다. 두 끝값이 각각
#: 무엇을 잘하고 못하는지가 분명해서(global은 LTM, zscore는 계층 혼합)
#: 중간값이 의미를 갖는다. 실측 곡선은 05/07 문서에 있다.
DEFAULT_HYBRID_W = 0.5


def hybrid_weight() -> float:
    try:
        w = float(os.environ.get("PREFETCH_HYBRID_W", DEFAULT_HYBRID_W))
    except ValueError:
        return DEFAULT_HYBRID_W
    return min(1.0, max(0.0, w))


def normalize_mode() -> str:
    name = os.environ.get("PREFETCH_NORMALIZE", DEFAULT_NORMALIZE).strip().lower()
    return name if name in {"global", "tier", "rrf", "zscore", "hybrid"} else DEFAULT_NORMALIZE


#: RRF 완충 상수. 클수록 순위 차이가 완만해진다. 관례값은 60이다.
RRF_K = 60
