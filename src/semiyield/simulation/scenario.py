"""Validated parameters for a deterministic three-stage simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    seed: int = 42
    batches: int = 100
    units_per_batch: int = 100

    def validate(self) -> None:
        if self.batches < 2 or self.units_per_batch < 1:
            raise ValueError("At least 2 batches and 1 unit per batch are required")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
