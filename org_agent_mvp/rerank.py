"""재정렬기 (cross-encoder). 1차 검색 상위 N개 청크의 순서만 다시 매긴다.

    RETRIEVER_RERANK=none                 끔 (기본)
    RETRIEVER_RERANK=<HF 모델 이름>        예: BAAI/bge-reranker-v2-m3
    RETRIEVER_RERANK_TOP_N=30             다시 매길 후보 수
    RETRIEVER_RERANK_MAX_LENGTH=512       질문+청크 최대 토큰

## 왜 순서만 바꾸나

cross-encoder 점수는 모델마다 눈금이 다르다(로짓, 0~1 확률 등). 그대로 쓰면
prefetch가 계층 점수를 합칠 때 LTM만 튀거나 눌린다(임베딩 RRF 때 겪은 문제, ltm_corpus 1-b 참고).
그래서 상위 N개가 원래 갖던 점수 값들은 그대로 두고, 그 값을 재정렬 순서대로 다시 나눠 준다.
점수 분포는 변하지 않고 순서만 바뀐다.

## 캐시

CPU에서 큰 모델은 질의 하나에 수 초가 걸린다. (모델, 질문, 청크 본문) 해시별 점수를
`cache/rerank/<모델 해시>.jsonl`에 쌓아 두고, 같은 쌍은 다시 계산하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

DEFAULT_TOP_N = 30
DEFAULT_MAX_LENGTH = 512
CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache" / "rerank"


class Scorer(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


def model_name() -> str:
    name = os.environ.get("RETRIEVER_RERANK", "none").strip()
    return "" if name.lower() in {"", "none", "off", "0"} else name


def top_n() -> int:
    try:
        return max(1, int(os.environ.get("RETRIEVER_RERANK_TOP_N", DEFAULT_TOP_N)))
    except ValueError:
        return DEFAULT_TOP_N


def _max_length() -> int:
    try:
        return max(32, int(os.environ.get("RETRIEVER_RERANK_MAX_LENGTH", DEFAULT_MAX_LENGTH)))
    except ValueError:
        return DEFAULT_MAX_LENGTH


def _batch_size() -> int:
    """CPU에서 한 번에 채점할 쌍 수. 큰 모델(약 5.7억)에 512토큰 쌍 16개를 묶으면
    중간 계산이 수 GB라 다른 프로세스와 함께 돌 때 메모리 부족으로 죽었다(2026-09-15). 결과 순서와는 무관하다."""
    try:
        return max(1, int(os.environ.get("RETRIEVER_RERANK_BATCH", 4)))
    except ValueError:
        return 4


def _pair_key(query: str, passage: str) -> str:
    return hashlib.blake2b(f"{query}\x00{passage}".encode("utf-8"), digest_size=12).hexdigest()


class CrossEncoderScorer:
    """sentence-transformers CrossEncoder + 디스크 캐시. 모델은 처음 점수를 낼 때 읽는다."""

    def __init__(self, name: str, max_length: int = DEFAULT_MAX_LENGTH):
        self.name = name
        self.max_length = max_length
        self._model = None
        digest = hashlib.blake2b(f"{name}|{max_length}".encode("utf-8"), digest_size=8).hexdigest()
        self.cache_path = CACHE_ROOT / f"{digest}.jsonl"
        self._cache: dict[str, float] = {}
        if self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        key, value = json.loads(line)
                        self._cache[key] = value

    def score(self, query: str, passages: list[str]) -> list[float]:
        keys = [_pair_key(query, p) for p in passages]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        if missing:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.name, max_length=self.max_length, device="cpu")
            values = self._model.predict([(query, passages[i]) for i in missing], batch_size=_batch_size(),
                                         show_progress_bar=False)
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                for i, value in zip(missing, values):
                    self._cache[keys[i]] = float(value)
                    fh.write(json.dumps([keys[i], float(value)]) + "\n")
        return [self._cache[k] for k in keys]


_LOADED: dict[tuple[str, int], Scorer] = {}


def get_scorer() -> Scorer | None:
    """설정된 재정렬기. 끄면 None. 같은 설정이면 프로세스 안에서 한 번만 만든다."""
    name = model_name()
    if not name:
        return None
    key = (name, _max_length())
    if key not in _LOADED:
        _LOADED[key] = CrossEncoderScorer(name, key[1])
    return _LOADED[key]


def reorder(scores: dict[int, float], query: str, passage_of, scorer: Scorer, n: int) -> dict[int, float]:
    """상위 n개 청크의 점수 값은 그대로 두고, 재정렬 점수 순서대로 다시 나눠 준다.

    동점(재정렬 점수가 같음)은 원래 순서를 유지한다. 나머지 청크는 건드리지 않는다.
    """
    head = sorted(scores, key=lambda idx: -scores[idx])[:n]
    if len(head) < 2:
        return scores
    values = [scores[idx] for idx in head]                      # 이미 내림차순
    rerank = scorer.score(query, [passage_of(idx) for idx in head])
    order = sorted(range(len(head)), key=lambda i: -rerank[i])  # 안정 정렬: 동점은 원래 순서
    out = dict(scores)
    for value, i in zip(values, order):
        out[head[i]] = value
    return out
