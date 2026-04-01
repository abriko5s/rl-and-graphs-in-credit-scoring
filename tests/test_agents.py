import numpy as np

from credit_scoring.agents.heuristic_agent import (
    AlwaysApproveAgent,
    AlwaysRejectAgent,
    EquityThresholdAgent,
)
from credit_scoring.config import Config
from credit_scoring.env import FlatCreditScoringEnv


def test_always_approve():
    agent = AlwaysApproveAgent()
    assert agent.predict(np.zeros(10)) == 1


def test_always_reject():
    agent = AlwaysRejectAgent()
    assert agent.predict(np.zeros(10)) == 0


def test_equity_threshold():
    config = Config(num_companies=10, num_steps=10)
    env = FlatCreditScoringEnv(config=config, seed=42)
    env.reset(seed=42)

    agent = EquityThresholdAgent(threshold=50.0)
    action = agent.predict(np.zeros(10), sim=env.sim)
    assert action in (0, 1)
