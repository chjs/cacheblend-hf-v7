"""LoadingController — pipelined KV loading + recompute ratio decisions.

Phase 4 implementation.

Design:
- StorageProfile enum (RAM/NVMe/SATA_SSD/SLOW_DISK) carries a relative
  loading-cost factor. Slower storage → loading dominates → recompute is
  comparatively cheaper → controller raises recompute_ratio.
- decide_recompute_ratio(storage_profile, base_ratio) is **monotone non-decreasing**
  in the storage slowness rank.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StorageProfile(Enum):
    """Per-tier relative load cost factor (RAM = 1.0 baseline).

    Numbers are illustrative — paper §6 mentions ~10× difference between RAM
    and SSD; we pick conservative monotone values that span 1×~50× range.
    """
    RAM = 1.0
    NVME = 4.0
    SATA_SSD = 12.0
    SLOW_DISK = 50.0

    @property
    def slowness_rank(self) -> int:
        """0 = fastest, larger = slower. Used for monotone checks."""
        return {
            StorageProfile.RAM: 0,
            StorageProfile.NVME: 1,
            StorageProfile.SATA_SSD: 2,
            StorageProfile.SLOW_DISK: 3,
        }[self]


@dataclass
class LoadingDecision:
    storage_profile: StorageProfile
    base_ratio: float
    recompute_ratio: float
    detail: str


class LoadingController:
    """Cost-aware ratio adjustment.

    Slow storage → loading dominates → recompute ratio rises (more tokens
    recomputed locally, fewer KV reads from slow tier). Fast storage (RAM)
    leaves base_ratio untouched.
    """

    # How aggressively each tier scales the base ratio above. Tuned so that
    # SLOW_DISK pushes a 0.15 ratio toward ~0.50 (the boundary where
    # recompute starts to dominate quality gain anyway).
    _MULTIPLIERS = {
        StorageProfile.RAM: 1.00,
        StorageProfile.NVME: 1.30,
        StorageProfile.SATA_SSD: 1.80,
        StorageProfile.SLOW_DISK: 3.30,
    }

    def __init__(self, max_ratio: float = 0.95):
        self.max_ratio = max_ratio  # ratio>=1 boundary safe-shortcut handles ≥1 anyway

    def decide_recompute_ratio(
        self,
        storage_profile: StorageProfile,
        base_ratio: float,
    ) -> LoadingDecision:
        """Return (and report) the ratio to use for this storage tier."""
        if not (0.0 <= base_ratio <= 1.0):
            raise ValueError(f"base_ratio must be in [0, 1], got {base_ratio}")

        m = self._MULTIPLIERS[storage_profile]
        scaled = base_ratio * m
        ratio = min(scaled, self.max_ratio)

        return LoadingDecision(
            storage_profile=storage_profile,
            base_ratio=base_ratio,
            recompute_ratio=ratio,
            detail=(
                f"profile={storage_profile.name} (slowness_rank={storage_profile.slowness_rank}), "
                f"base={base_ratio:.3f}, multiplier={m:.2f}, "
                f"scaled={scaled:.3f}, capped@{self.max_ratio:.2f} → {ratio:.3f}"
            ),
        )
