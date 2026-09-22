from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import numpy as np


class Provenance(str, Enum):
    EMPIRICAL = "EMPIRICAL"
    MODEL_INPUT = "MODEL_INPUT"
    DERIVED = "DERIVED"
    HYPOTHESIS = "HYPOTHESIS"
    NUMERICAL = "NUMERICAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class PYMState:
    position: np.ndarray
    momentum: np.ndarray
    mass: float = 1.0
    charge: float = 0.0
    spin: float = 0.5
    interaction_signal: float = 0.0
    memory: float = 0.0
    lambda_pym: float = 0.0
    alpha: float = 0.95
    provenance_map: dict[str, Provenance] = field(default_factory=lambda: {
        "position": Provenance.MODEL_INPUT,
        "momentum": Provenance.MODEL_INPUT,
        "mass": Provenance.MODEL_INPUT,
        "charge": Provenance.MODEL_INPUT,
        "spin": Provenance.MODEL_INPUT,
        "interaction_signal": Provenance.HYPOTHESIS,
        "memory": Provenance.HYPOTHESIS,
        "lambda_pym": Provenance.HYPOTHESIS,
        "alpha": Provenance.HYPOTHESIS,
    })

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).copy()
        self.momentum = np.asarray(self.momentum, dtype=float).copy()
        if self.position.shape != self.momentum.shape:
            raise ValueError("position and momentum must have identical shapes")
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

    def clone(self) -> "PYMState":
        return replace(
            self,
            position=self.position.copy(),
            momentum=self.momentum.copy(),
            provenance_map=self.provenance_map.copy(),
        )
