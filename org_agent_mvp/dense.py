"""3단계: 임베딩 검색. sparse(BM25)와 RRF로 합친다.

    RETRIEVER_DENSE=1              켠다
    RETRIEVER_DENSE_MODEL=<모델>    기본 intfloat/multilingual-e5-small

## 왜 필요한가

BM25까지 올려도 **어휘가 겹치지 않으면 못 찾는다.**
평가셋에서 실제로 막힌 사례다.

    질의   "영상 송수신에 어떤 프로토콜을 쓰기로 했어?"
    문서   "... WebRTC(Web Real-Time Communications)를 활용하여 영상/음성 정보를 ..."

"프로토콜"과 "WebRTC"는 형태소로 잘라도 BM25로 재도 만나지 않는다.
의미가 가까우면 벡터도 가깝다는 성질이 필요하다.

## 왜 여기서 RRF를 쓰나

dense 점수(코사인, 0~1)와 sparse 점수(BM25, 0~30+)는 눈금이 다르다.
정규화는 질의마다 최대값이 달라 불안정하다. 순위는 눈금이 없어서 안전하다.

    score(청크) = Σ  1 / (k + 그 검색기에서의 순위)
                검색기

계층 병합에는 RRF를 쓰지 않는다. 거기서는 오히려 나빴다 —
계층 1등이 모두 동률이 되어 tier_prior가 순위를 독점했고,
실측으로 LTM MRR이 0.450에서 0.110으로 떨어졌다.
같은 코퍼스 안에서 두 검색기를 합칠 때만 쓴다.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

#: 기본 모델. 118M으로 작아 CPU에서도 돈다.
#: doc_rag는 mE5-base(278M)를 쓰는데, 여기서는 재현 비용을 우선했다.
DEFAULT_MODEL = "intfloat/multilingual-e5-small"

#: E5 계열은 질의와 문서에 서로 다른 접두어를 붙여야 성능이 나온다.
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

#: RRF 완충 상수.
RRF_K = 60

#: dense 후보를 몇 건까지 볼지. sparse와 같은 수로 맞춘다.
TOP_N = 200

#: 인코딩할 최대 토큰 수.
#:
#: 모델 기본값은 512인데 CPU에서는 그게 그대로 시간이 된다. 실측이다.
#:
#:     max_seq=512   26,031청크 99분
#:     max_seq=256   26,031청크 53분   <- 채택
#:     max_seq=192   26,031청크 40분
#:
#: 청크 평균이 754자(대략 350~450 서브워드)라 256이면 앞 절반쯤만 들어간다.
#: 검색에서는 앞부분(제목·도입)이 더 중요하고 제목을 맨 앞에 붙여 두었으므로
#: 손실을 감수했다. GPU가 있으면 512로 되돌리는 게 맞다.
MAX_SEQ_LENGTH = 256

#: 인코딩 배치. CPU에서는 크게 잡는 편이 낫다.
BATCH_SIZE = 64


def enabled() -> bool:
    return os.environ.get("RETRIEVER_DENSE", "0").strip() in {"1", "true", "yes"}


def model_name() -> str:
    return os.environ.get("RETRIEVER_DENSE_MODEL", DEFAULT_MODEL).strip()


def cache_path(corpus_path: Path, n_chunks: int) -> Path:
    """임베딩 캐시 경로.

    코퍼스 파일 크기·수정시각과 모델 이름으로 키를 만든다.
    코퍼스가 바뀌면 자동으로 다른 파일이 되므로 낡은 벡터를 쓸 일이 없다.
    """
    stat = corpus_path.stat()
    key = f"{corpus_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{n_chunks}|{model_name()}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    root = Path(__file__).resolve().parents[1] / "cache" / "dense"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{digest}.npy"


class DenseIndex:
    """청크 임베딩과 코사인 검색.

    모델 로딩과 인코딩이 무거워서, 벡터를 만들고 나면 `.npy`로 떨궈 둔다.
    두 번째 실행부터는 로딩만 하면 된다.
    """

    def __init__(self, corpus_path: Path, texts: list[str], verbose: bool = True):
        self.path = cache_path(corpus_path, len(texts))
        self._model = None
        if self.path.exists():
            self.vectors = np.load(self.path)
            if self.vectors.shape[0] != len(texts):
                # 캐시 키에 청크 수가 들어가므로 보통 안 걸린다. 방어선이다.
                self.vectors = self._encode_all(texts, verbose)
        else:
            self.vectors = self._encode_all(texts, verbose)

    # ------------------------------------------------------------------ build

    def _load_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # 기본값이 코어 수의 절반이라 CPU에서 손해다.
            torch.set_num_threads(os.cpu_count() or 8)
            self._model = SentenceTransformer(model_name())
            self._model.max_seq_length = MAX_SEQ_LENGTH
        return self._model

    def _encode_all(self, texts: list[str], verbose: bool) -> np.ndarray:
        model = self._load_model()
        if verbose:
            print(f"[dense] {len(texts):,}청크 인코딩 시작 ({model_name()})", flush=True)
        vectors = model.encode(
            [PASSAGE_PREFIX + t for t in texts],
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,      # 정규화해 두면 내적이 곧 코사인이다
            show_progress_bar=verbose,
        ).astype(np.float32)
        np.save(self.path, vectors)
        if verbose:
            print(f"[dense] 저장: {self.path}", flush=True)
        return vectors

    # ----------------------------------------------------------------- search

    def top_n(self, query: str, n: int = TOP_N) -> list[int]:
        """코사인 상위 n개 청크 번호. 벡터가 정규화돼 있어 내적으로 충분하다."""
        model = self._load_model()
        q = model.encode(
            [QUERY_PREFIX + query], convert_to_numpy=True, normalize_embeddings=True
        )[0].astype(np.float32)
        sims = self.vectors @ q
        n = min(n, sims.shape[0])
        idx = np.argpartition(-sims, kth=n - 1)[:n]
        return [int(i) for i in idx[np.argsort(-sims[idx])]]


def dense_weight() -> float:
    """dense 순위에 줄 비중. 1.0이면 sparse와 대등하다.

    낮추면 sparse 순위를 더 존중한다. 어휘가 정확히 겹치는 질의를
    dense가 밀어내는 것을 막는 용도다.

    기본값 0.5는 실측으로 골랐다. LTM 검색만 재서

        w=1.0   재현율 41.2%  MRR 0.467  전멸 4
        w=0.5   재현율 46.8%  MRR 0.469  전멸 2   <- 채택
        w=0.3   재현율 44.9%  MRR 0.457  전멸 4

    이 평가셋의 질의는 대부분 정확한 사실 조회다("443,750천원", "SM-LowV-001").
    그런 질의는 어휘 일치가 유리해서 dense를 대등하게 두면 손해가 난다.
    """
    try:
        return float(os.environ.get("RETRIEVER_DENSE_WEIGHT", "0.5"))
    except ValueError:
        return 1.0


def rrf_merge(sparse_ranked: list[int], dense_ranked: list[int],
              k: int = RRF_K, w_dense: float | None = None) -> dict[int, float]:
    """두 순위 목록을 순위 기반으로 합친다.

    점수를 안 쓰므로 BM25와 코사인의 눈금 차이가 문제되지 않는다.
    양쪽에서 다 상위인 청크가 이기고, 한쪽에서만 잡힌 것도 완전히 밀리지는 않는다.

    `w_dense`로 dense 쪽 비중을 조절한다. 1.0이면 표준 RRF다.
    """
    if w_dense is None:
        w_dense = dense_weight()
    out: dict[int, float] = {}
    for ranked, weight in ((sparse_ranked, 1.0), (dense_ranked, w_dense)):
        for rank, idx in enumerate(ranked, start=1):
            out[idx] = out.get(idx, 0.0) + weight / (k + rank)
    return out
