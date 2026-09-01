from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_store import MemoryStore
from .query_analyzer import QueryPlan


TIERS = ("stm", "mtm", "ltm")


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

        query = plan.query_rewrites[-1]
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

        # 2. evidence_id 기준 중복 제거. 원점수가 높은 쪽을 남긴다.
        deduped: dict[str, dict[str, Any]] = {}
        for card in collected:
            evidence_id = str(card["evidence_id"])
            current = deduped.get(evidence_id)
            if current is None or self._raw(card) > self._raw(current):
                deduped[evidence_id] = card
        candidates = list(deduped.values())

        # 3. 전역 정규화. tier마다 점수 스케일이 다르므로 비교 전에 맞춘다.
        max_raw = max((self._raw(card) for card in candidates), default=0.0)
        denominator = max_raw or 1.0
        for card in candidates:
            normalized = self._raw(card) / denominator
            card["normalized_score"] = round(normalized, 4)
            # 가산이 아니라 승산이다. 관련성 0인 문서는 tier와 무관하게 0으로 남는다.
            card["final_score"] = round(
                normalized * (1.0 + self.alpha * card["tier_prior"]), 4
            )

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
