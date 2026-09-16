from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from .memory_store import MemoryStore
from .scoring import RRF_K, ZSCORE_MIN_POOL, hybrid_weight, normalize_mode
from .query_analyzer import QueryPlan


TIERS = ("stm", "mtm", "ltm")


def prefetch_query(plan: QueryPlan) -> str:
    """계층 검색에 실제로 들어가는 질의 문자열.

    rewrite 중 하나만 고르면 정보가 빠진다. LLM이 여러 개를 만들 때
    마지막이 가장 짧고 정보가 적은 경우가 실제로 있었다.
    원 질문이 항상 포함되도록 전부 이어 붙인다.

    함수로 빼 둔 이유는, 이 문자열을 만드는 곳이 세 군데였고 서로 달랐기
    때문이다 — prefetch는 전부 이어 붙였는데 run_eval의 quota baseline과
    agent_runtime의 로그는 `query_rewrites[-1]`을 썼다. 앞의 것은 비교
    실험에 질의 구성 차이를 섞었고, 뒤의 것은 로그에 실제로 검색하지 않은
    질의를 남겼다. 한 곳에서만 만들면 다시 어긋나지 않는다.
    """
    return " ".join(dict.fromkeys(plan.query_rewrites))


@dataclass(frozen=True)
class PrefetchResult:
    """A방식 결과.

    tier 가중치는 검색 자리 수(쿼터)가 아니라 순위 점수의 prior로만 쓴다.
    후보를 tier별로 미리 자르지 않고 넓게 모은 뒤, 정규화한 관련성 점수에
    tier prior를 곱해 한 번에 순위를 매기고 상대 임계값으로 자른다.
    """

    tier_priors: dict[str, float]
    pool_per_tier: int
    cards: list[dict[str, Any]]
    tier_result_counts: dict[str, int]
    collected_count: int
    deduped_count: int
    max_raw_score: float
    cut_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier_priors": self.tier_priors,
            "pool_per_tier": self.pool_per_tier,
            "cards": self.cards,
            "tier_result_counts": self.tier_result_counts,
            "collected_count": self.collected_count,
            "deduped_count": self.deduped_count,
            "max_raw_score": self.max_raw_score,
            "cut_threshold": self.cut_threshold,
        }


class MemoryPrefetcher:
    def __init__(
        self,
        memory_store: MemoryStore,
        total_top_k: int = 8,
        *,
        alpha: float = 1.0,
        pool_per_tier: int = 20,
        cut_ratio: float = 0.3,
        min_cards: int = 1,
        tier_floor: int = 1,
    ):
        self.memory_store = memory_store
        self.total_top_k = total_top_k
        self.alpha = alpha
        self.pool_per_tier = pool_per_tier
        self.cut_ratio = cut_ratio
        self.min_cards = min_cards
        self.tier_floor = tier_floor

    def prefetch(self, plan: QueryPlan) -> PrefetchResult:
        priors = {tier: float(plan.memory_weights.get(tier, 0.0)) for tier in TIERS}
        if not plan.memory_needed:
            return PrefetchResult(
                tier_priors=priors,
                pool_per_tier=0,
                cards=[],
                tier_result_counts={tier: 0 for tier in TIERS},
                collected_count=0,
                deduped_count=0,
                max_raw_score=0.0,
                cut_threshold=0.0,
            )

        query = prefetch_query(plan)
        collected: list[dict[str, Any]] = []
        tier_result_counts = {tier: 0 for tier in TIERS}

        # 1. tier별로 자르지 않고 넓게 수집한다.
        for tier in TIERS:
            result = self.memory_store.retrieve(
                tier=tier,
                query=query,
                filters=plan.filters,
                top_k=self.pool_per_tier,
            )
            tier_result_counts[tier] = int(result.get("result_count", 0))
            for card in result["results"]:
                card = dict(card)
                card["tier_prior"] = priors[tier]
                collected.append(card)

        # 2. 중복 제거. 원점수가 높은 쪽을 남기고, 접힌 쪽은 카드에 기록한다.
        deduped: dict[str, dict[str, Any]] = {}
        for card in collected:
            key = _dedupe_key(card)
            current = deduped.get(key)
            if current is None:
                deduped[key] = card
            elif self._raw(card) > self._raw(current):
                deduped[key] = _fold_into(card, current)
            else:
                deduped[key] = _fold_into(current, card)
        candidates = list(deduped.values())

        # 3. 정규화. 어느 기준으로 맞출지는 설정으로 고른다(PREFETCH_NORMALIZE).
        #
        #    global (기본)  전역 최고점 하나로 나눈다.
        #                  계층 간 점수가 비교 가능하다는 전제가 필요하다.
        #    tier          계층별 최고점으로 나눈다.
        #                  각 계층 1등이 모두 1.0이 되므로 순위를 tier_prior가 정한다.
        #
        #    점수 함수가 BM25로 바뀌면 이 선택이 결과를 크게 흔든다.
        #    BM25의 IDF는 `ln(1 + (N-df+0.5)/(df+0.5))`인데 N이 계층마다 다르다 —
        #    STM/MTM은 문서 85건, LTM은 청크 26,031건이다.
        #
        #      freq   STM 8.9 / MTM 9.0 / LTM  9.6   비슷하다
        #      bm25   STM 9.0 / MTM 8.9 / LTM 20.5   LTM이 2배 이상
        #
        #    global을 쓰면 BM25에서 STM 최고점이 0.44로 눌려 cut_ratio(0.3)에 걸린다.
        #    반대로 tier를 쓰면 계층 1등끼리의 실제 점수 차이가 사라진다.
        #    어느 쪽도 무조건 옳지 않아서 재고 고르도록 열어 뒀다.
        mode = normalize_mode()
        max_raw = max((self._raw(card) for card in candidates), default=0.0)
        if mode == "rrf":
            # 점수 대신 **계층 안에서의 순위**로 맞춘다.
            #
            # 점수 눈금을 아예 안 쓰므로 계층마다 점수 함수가 달라도 섞인다.
            # BM25처럼 코퍼스 크기에 따라 눈금이 변하는 함수를 쓸 때 필요하다.
            #
            #     normalized = (k + 1) / (k + rank)     rank는 1부터
            #
            # k가 클수록 순위 차이가 완만해진다. RRF 관례대로 60을 쓴다.
            k = RRF_K
            by_tier: dict[str, list[dict[str, Any]]] = {}
            for card in candidates:
                by_tier.setdefault(str(card.get("tier", "")), []).append(card)
            ranks: dict[int, float] = {}
            for tier_cards in by_tier.values():
                tier_cards.sort(key=self._raw, reverse=True)
                for rank, card in enumerate(tier_cards, start=1):
                    ranks[id(card)] = (k + 1.0) / (k + rank)
            for card in candidates:
                card["normalized_score"] = round(ranks[id(card)], 4)
                card["final_score"] = round(
                    ranks[id(card)] * (1.0 + self.alpha * card["tier_prior"]), 4
                )
            return self._finish(candidates, priors, collected, deduped,
                                tier_result_counts, max_raw)

        if mode in {"zscore", "hybrid"}:
            # 계층 최고점이 아니라 **계층 안에서 몇 σ 튀는지**로 맞춘다.
            #
            # tier 모드는 각 계층 1등을 무조건 1.0으로 만들어서, 볼 것이 없는
            # 계층의 1등도 정답을 가진 계층의 1등과 같은 점수를 받는다.
            # 그래서 LTM 단독 질문에서 STM/MTM이 자리를 뺏었다(LTM MRR 0.451 -> 0.176).
            #
            # 표준화는 그 구분을 남긴다. 정답이 있는 계층은 top이 자기 풀 평균에서
            # 크게 벗어나고, 없는 계층은 조금밖에 안 벗어난다.
            # 계층 안에서는 선형변환이라 순위가 바뀌지 않는다.
            hybrid_w = hybrid_weight() if mode == "hybrid" else 1.0
            by_tier: dict[str, list[dict[str, Any]]] = {}
            for card in candidates:
                by_tier.setdefault(str(card.get("tier", "")), []).append(card)
            for tier_cards in by_tier.values():
                raws = [self._raw(card) for card in tier_cards]
                n = len(raws)
                mean = sum(raws) / n
                var = sum((r - mean) ** 2 for r in raws) / n
                sd = math.sqrt(var)
                for card in tier_cards:
                    if n < ZSCORE_MIN_POOL or sd <= 0.0:
                        # 표본이 적으면 평균·표준편차가 의미 없다.
                        # 그 계층만 global로 처리한다 — 눈금은 못 맞추지만
                        # 최소한 없는 근거를 지어내지 않는다.
                        card["normalized_score"] = round(
                            self._raw(card) / (max_raw or 1.0), 4
                        )
                    else:
                        z = (self._raw(card) - mean) / sd
                        norm = 1.0 / (1.0 + math.exp(-z))
                        if mode == "hybrid":
                            # zscore만 쓰면 **볼 것이 없는 계층도** 자기 풀 1등을
                            # 띄워 준다. tier 모드가 실패한 것과 같은 이유다.
                            # 풀이 top_k로 잘려 있어서 계층마다 분포 성격이 다른
                            # 것도 겹친다 — LTM의 top 20은 26,031청크에서 걸러진
                            # 정예라 자기들끼리 촘촘하고, STM의 top 20은 40문서 중
                            # 절반이라 성긴다. 큰 코퍼스를 뒤진 계층이 손해를 본다.
                            #
                            # 그래서 절대 점수(global)와 기하평균을 낸다.
                            # 두 신호가 **모두** 높아야 높아진다 —
                            # global이 "이 계층에 실제로 쓸 만한 게 있는가"를 지키고,
                            # z가 "계층 눈금 차이"를 걷어낸다.
                            g = self._raw(card) / (max_raw or 1.0)
                            norm = (g ** (1.0 - hybrid_w)) * (norm ** hybrid_w)
                        card["normalized_score"] = round(norm, 4)
                    card["final_score"] = round(
                        card["normalized_score"] * (1.0 + self.alpha * card["tier_prior"]),
                        4,
                    )
            return self._finish(candidates, priors, collected, deduped,
                                tier_result_counts, max_raw)

        if mode == "tier":
            bases: dict[str, float] = {}
            for card in candidates:
                key = str(card.get("tier", ""))
                bases[key] = max(bases.get(key, 0.0), self._raw(card))
        else:
            bases = {}
        for card in candidates:
            denominator = (
                bases.get(str(card.get("tier", ""))) if mode == "tier" else max_raw
            ) or 1.0
            normalized = self._raw(card) / denominator
            card["normalized_score"] = round(normalized, 4)
            # 가산이 아니라 승산이다. 관련성 0인 문서는 tier와 무관하게 0으로 남는다.
            card["final_score"] = round(
                normalized * (1.0 + self.alpha * card["tier_prior"]), 4
            )

        return self._finish(candidates, priors, collected, deduped,
                            tier_result_counts, max_raw)

    def _finish(
        self,
        candidates: list[dict[str, Any]],
        priors: dict[str, float],
        collected: list[dict[str, Any]],
        deduped: dict[str, dict[str, Any]],
        tier_result_counts: dict[str, int],
        max_raw: float,
    ) -> PrefetchResult:
        """정규화가 끝난 카드로 컷과 계층 최소 자리를 적용한다.

        정규화 방식(global / tier / rrf)이 갈라져서 뒷부분만 따로 뺐다.
        """
        ranked = sorted(candidates, key=lambda card: card["final_score"], reverse=True)

        # 4. 상대 임계값으로 자른다. 절대값은 코퍼스가 바뀌면 흔들린다.
        top_score = ranked[0]["final_score"] if ranked else 0.0
        cut_threshold = round(top_score * self.cut_ratio, 4)
        kept = [card for card in ranked if card["final_score"] >= cut_threshold]
        # 최고점 문서는 항상 컷을 통과하므로 후보가 있으면 kept는 비지 않는다.
        # min_cards는 컷을 우회하는 장치가 아니라 후보가 아예 없을 때를 위한 방어선이다.
        # 값을 2 이상으로 올리면 컷 아래 문서를 강제로 넣게 되므로 주의한다.
        if ranked and len(kept) < self.min_cards:
            kept = ranked[: self.min_cards]
        cards = self._apply_tier_floor(kept, priors)

        return PrefetchResult(
            tier_priors=priors,
            pool_per_tier=self.pool_per_tier,
            cards=cards,
            tier_result_counts=tier_result_counts,
            collected_count=len(collected),
            deduped_count=len(deduped),
            max_raw_score=round(max_raw, 3),
            cut_threshold=cut_threshold,
        )

    def _apply_tier_floor(
        self,
        eligible: list[dict[str, Any]],
        priors: dict[str, float],
    ) -> list[dict[str, Any]]:
        """계층별 최소 자리를 보장한다.

        코퍼스가 커지면 한 계층이 상위를 독점해 다른 계층의 정답이 밀려난다.
        문서 115건으로 측정했을 때 순수 전역 순위는 계층 쿼터 방식보다
        재현율이 낮았고, 원인은 계층 다양성 손실이었다.

        다만 옛 쿼터 방식처럼 자리를 억지로 채우지는 않는다.
        **이미 컷을 통과한 문서 중에서만** 계층별 최소 자리를 확보하므로,
        관련 없는 문서가 끌려 들어오지 않는다.
        """
        if self.tier_floor <= 0 or not eligible:
            return eligible[: self.total_top_k]

        reserved: list[dict[str, Any]] = []
        taken: set[str] = set()
        for tier in TIERS:
            if priors.get(tier, 0.0) <= 0:
                continue
            picked = 0
            for card in eligible:
                if picked >= self.tier_floor:
                    break
                if str(card.get("tier", "")).lower() != tier:
                    continue
                reserved.append(card)
                taken.add(str(card["evidence_id"]))
                picked += 1

        # 예약분을 앞에 두어야 상한(total_top_k)에서 잘리지 않는다.
        # 여기서 다시 전역 정렬하면 예약이 무효가 되므로 자른 뒤에 정렬한다.
        rest = [card for card in eligible if str(card["evidence_id"]) not in taken]
        selected = (reserved + rest)[: self.total_top_k]
        return sorted(selected, key=lambda card: card["final_score"], reverse=True)

    @staticmethod
    def _raw(card: dict[str, Any]) -> float:
        return float(card.get("retrieval_score", 0.0))


# ------------------------------------------------------- 계층 간 중복 제거
#
# 같은 내용이 계층마다 따로 올라오는 문제가 있다. 원본 문서(LTM)를 발췌·정리해
# 노트(MTM)로 옮기면 id가 달라서 두 장이 각각 후보가 된다. 근거 카드는 8장뿐이라
# 같은 내용이 두 자리를 먹고, 그 문서를 묻는 **다른 질문**에서는 노트가 원본을 밀어낸다
# (2026-09-16 측정: 같은 문서의 다른 질문 156개에서 문서 MRR 0.904 -> 0.780).
#
# 그래서 카드가 가리키는 **출처**가 같으면 한 장으로 접는다. 판별은 문서가 스스로 밝힌
# 출처(`source_document` 머리말)로 한다. 내용 비교나 임베딩은 쓰지 않는다 — 기준이 모호해진다.
#
# 접을 때 버리지 않는다. 점수가 높은 쪽을 세우고 접힌 쪽은 `merged_evidence_ids`에 남긴다.
# (LTM 안의 버전 접기가 쓰는 `folded_*`와는 다른 키다. 그쪽은 같은 계층의 사본 묶음이다.)


def merge_same_source() -> bool:
    """PREFETCH_MERGE_SAME_SOURCE=0이면 예전처럼 evidence_id로만 중복을 본다."""
    return os.environ.get("PREFETCH_MERGE_SAME_SOURCE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _dedupe_key(card: dict[str, Any]) -> str:
    """중복 판별 키. 출처를 알 수 있으면 출처, 아니면 evidence_id다.

    LTM 카드는 문서 단위라 `source_ref.document_id`가 곧 출처다.
    시드 문서는 머리말에 `source_document`를 적었을 때만 출처가 생긴다 — 안 적으면 예전과 같다.
    """
    ref = card.get("source_ref") or {}
    if merge_same_source():
        origin = ref.get("derived_from") or {}
        source_id = origin.get("document_id")
        if not source_id and str(card.get("tier", "")).upper() == "LTM":
            source_id = ref.get("document_id")
        if source_id:
            return f"src:{source_id}"
    return str(card.get("evidence_id", ""))


def _fold_into(keep: dict[str, Any], dropped: dict[str, Any]) -> dict[str, Any]:
    """`dropped`를 `keep`에 접어 넣은 새 카드를 돌려준다. 원본 두 장은 건드리지 않는다."""
    card = dict(keep)
    ref = dict(card.get("source_ref") or {})
    dropped_ref = dropped.get("source_ref") or {}
    ids = [*ref.get("merged_evidence_ids", []), str(dropped.get("evidence_id", "")),
           *dropped_ref.get("merged_evidence_ids", [])]
    titles = [*ref.get("merged_titles", []), str(dropped.get("title", "")),
              *dropped_ref.get("merged_titles", [])]
    tiers = [*ref.get("merged_tiers", []), str(dropped.get("tier", "")), *dropped_ref.get("merged_tiers", [])]
    ref["merged_evidence_ids"] = ids
    ref["merged_titles"] = titles
    ref["merged_tiers"] = tiers
    ref["merged_count"] = len(ids)
    card["source_ref"] = ref
    return card
