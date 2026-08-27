from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory_store import MemoryStore
from .query_analyzer import QueryPlan


@dataclass(frozen=True)
class PrefetchResult:
    allocations: dict[str, int]
    cards: list[dict[str, Any]]
    tier_result_counts: dict[str, int]
    collected_count: int
    deduped_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocations": self.allocations,
            "cards": self.cards,
            "tier_result_counts": self.tier_result_counts,
            "collected_count": self.collected_count,
            "deduped_count": self.deduped_count,
        }


class MemoryPrefetcher:
    def __init__(self, memory_store: MemoryStore, total_top_k: int = 8):
        self.memory_store = memory_store
        self.total_top_k = total_top_k

    def prefetch(self, plan: QueryPlan) -> PrefetchResult:
        if not plan.memory_needed:
            return PrefetchResult(
                allocations={"stm": 0, "mtm": 0, "ltm": 0},
                cards=[],
                tier_result_counts={"stm": 0, "mtm": 0, "ltm": 0},
                collected_count=0,
                deduped_count=0,
            )

        allocations = self._allocate(plan.memory_weights)
        query = plan.query_rewrites[-1]
        collected: list[dict[str, Any]] = []
        tier_result_counts = {"stm": 0, "mtm": 0, "ltm": 0}
        for tier in ("stm", "mtm", "ltm"):
            top_k = allocations[tier]
            if top_k == 0:
                continue
            result = self.memory_store.retrieve(
                tier=tier,
                query=query,
                filters=plan.filters,
                top_k=top_k,
            )
            tier_result_counts[tier] = int(result.get("result_count", 0))
            for card in result["results"]:
                card = dict(card)
                card["prefetch_tier_weight"] = plan.memory_weights[tier]
                card["rerank_score"] = round(
                    float(card.get("retrieval_score", 0.0))
                    + (plan.memory_weights[tier] * 5.0),
                    3,
                )
                collected.append(card)

        deduped: dict[str, dict[str, Any]] = {}
        for card in collected:
            evidence_id = str(card["evidence_id"])
            if (
                evidence_id not in deduped
                or card["rerank_score"] > deduped[evidence_id]["rerank_score"]
            ):
                deduped[evidence_id] = card
        cards = sorted(deduped.values(), key=lambda item: item["rerank_score"], reverse=True)
        return PrefetchResult(
            allocations=allocations,
            cards=cards[: self.total_top_k],
            tier_result_counts=tier_result_counts,
            collected_count=len(collected),
            deduped_count=len(deduped),
        )

    def _allocate(self, weights: dict[str, float]) -> dict[str, int]:
        tiers = ("stm", "mtm", "ltm")
        raw = {tier: max(0.0, weights.get(tier, 0.0)) * self.total_top_k for tier in tiers}
        allocation = {tier: int(raw[tier]) for tier in tiers}
        remaining = self.total_top_k - sum(allocation.values())
        order = sorted(tiers, key=lambda tier: raw[tier] - allocation[tier], reverse=True)
        for tier in order[:remaining]:
            allocation[tier] += 1
        for tier in tiers:
            if weights.get(tier, 0.0) > 0 and allocation[tier] == 0:
                donor = max(tiers, key=lambda item: allocation[item])
                if allocation[donor] > 1:
                    allocation[donor] -= 1
                    allocation[tier] = 1
        return allocation
