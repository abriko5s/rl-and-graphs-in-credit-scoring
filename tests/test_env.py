import numpy as np

from credit_scoring.config import Config
from credit_scoring.env import CreditScoringEnv, FlatCreditScoringEnv


def _make_config() -> Config:
    return Config(num_companies=10, num_steps=20)


def test_credit_scoring_env_reset():
    env = CreditScoringEnv(config=_make_config(), seed=42)
    obs, info = env.reset(seed=42)

    assert "node_features" in obs
    assert "edge_index" in obs
    assert "application_features" in obs
    assert obs["node_features"].shape == (10, 7)
    assert obs["application_features"].shape == (7,)


def test_credit_scoring_env_step():
    env = CreditScoringEnv(config=_make_config(), seed=42)
    obs, _ = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(1)

    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cumulative_pnl" in info


def test_credit_scoring_env_full_episode():
    env = CreditScoringEnv(config=_make_config(), seed=42)
    obs, _ = env.reset(seed=42)
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    assert steps <= 20


def test_flat_env_reset():
    env = FlatCreditScoringEnv(config=_make_config(), seed=42)
    obs, info = env.reset(seed=42)

    assert obs.shape == (11,)
    assert obs.dtype == np.float32


def test_flat_env_step():
    env = FlatCreditScoringEnv(config=_make_config(), seed=42)
    obs, _ = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == (11,)
    assert isinstance(reward, float)


def test_flat_env_full_episode():
    env = FlatCreditScoringEnv(config=_make_config(), seed=42)
    obs, _ = env.reset(seed=42)
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    assert isinstance(total_reward, float)


def test_env_reproducibility():
    config = _make_config()

    env1 = CreditScoringEnv(config=config, seed=42)
    obs1, _ = env1.reset(seed=42)

    env2 = CreditScoringEnv(config=config, seed=42)
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_array_equal(obs1["node_features"], obs2["node_features"])
    np.testing.assert_array_equal(obs1["application_features"], obs2["application_features"])
