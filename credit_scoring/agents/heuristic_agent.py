"""
Some Heuristics: Always Approve, Always Reject, Equity Threshold
Kinda baselines
"""

from __future__ import annotations

import numpy as np

from credit_scoring.market import MarketSimulator


class AlwaysApproveAgent:
    """
    Одобряет все заявки
    """

    def predict(self, obs: dict | np.ndarray, sim: MarketSimulator | None = None) -> int:
        return 1


class AlwaysRejectAgent:
    """
    Отклоняет все заявки
    """

    def predict(self, obs: dict | np.ndarray, sim: MarketSimulator | None = None) -> int:
        return 0


class EquityThresholdAgent:
    """
    Одобряет заявку, если equity заявителя превышает threshold (гп но на него все равно пока что)
    """

    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold

    def predict(self, obs: dict | np.ndarray, sim: MarketSimulator | None = None) -> int:
        if sim is None:  # выпадало, перестраховался
            raise ValueError("EquityThresholdAgent requires access to sim")

        app = sim.current_application
        company = sim.companies[app.applicant_id]
        return 1 if company.equity() > self.threshold else 0
