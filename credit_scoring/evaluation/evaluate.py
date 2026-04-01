"""
Evaluating agents.
Passes the agent through  N episodes with different seeds,
collects metrics (P&L, default rate, approval rate by sector)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3 import PPO

from credit_scoring.config import Config
from credit_scoring.env import CreditScoringEnv, FlatCreditScoringEnv


def evaluate_rl_agent(
    model: PPO,
    config: Config,
    num_episodes: int = 10,
    seed: int = 0,
    use_flat_env: bool = False,
) -> dict[str, Any]:
    """
    Оценить RL-агента (MLP или GNN) на нескольких эпизодах

    Args:
        model: обученная модель
        config: конфигурация
        num_episodes: количество эпизодов
        seed: начальный seed
        use_flat_env: True для MLP

    Returns:
        метрики
    """
    results = _empty_results()

    for ep in range(num_episodes):
        ep_seed = seed + ep
        if use_flat_env:
            env = FlatCreditScoringEnv(config=config, seed=ep_seed)
        else:
            env = CreditScoringEnv(config=config, seed=ep_seed)

        obs, _ = env.reset(seed=ep_seed)
        done = False
        pnl_history = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            pnl_history.append(reward)
            done = terminated or truncated

        sim = env.sim if hasattr(env, "sim") else env._full_env.sim

        # adding cooldown
        cooldown_steps = len(sim.step_pnl_history) - len(pnl_history)
        if cooldown_steps > 0:
            pnl_history.extend(sim.step_pnl_history[-cooldown_steps:])

        _collect_episode_results(results, pnl_history, sim)

    return _aggregate_results(results, num_episodes)


def evaluate_heuristic_agent(
    agent,
    config: Config,
    num_episodes: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Не RL-агенты

    Args:
        agent: объект с методом predict(obs, sim=...)
        config: конфигурация
        num_episodes: количество эпизодов
        seed: начальный seed

    Returns:
        метрики
    """
    results = _empty_results()

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = FlatCreditScoringEnv(config=config, seed=ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        done = False
        pnl_history = []

        if hasattr(agent, "reset"):
            agent.reset()

        while not done:
            action = agent.predict(obs, sim=env.sim)
            obs, reward, terminated, truncated, info = env.step(int(action))
            pnl_history.append(reward)
            done = terminated or truncated

        # adding cooldown
        cooldown_steps = len(env.sim.step_pnl_history) - len(pnl_history)
        if cooldown_steps > 0:
            pnl_history.extend(env.sim.step_pnl_history[-cooldown_steps:])

        _collect_episode_results(results, pnl_history, env.sim)

    return _aggregate_results(results, num_episodes)


def _empty_results() -> dict:
    return {
        "cumulative_pnls": [],
        "total_defaults": [],
        "approval_rates": [],
        "borrower_default_rates": [],
        "pnl_histories": [],
        "sector_approvals": [],
        "sector_requests": [],
        "sector_pnls": [],
        # abs
        "sharpe_ratios": [],
        "roas": [],
        "survival_rates": [],
        "max_cascade_depths": [],
        # relative
        "pnl_per_step": [],
        "pnl_per_company": [],
    }


def _collect_episode_results(
    results: dict, pnl_history: list[float], sim
) -> None:
    metrics = sim.get_metrics()
    results["cumulative_pnls"].append(metrics["cumulative_pnl"])
    results["total_defaults"].append(metrics["total_defaults"])
    results["approval_rates"].append(metrics["approval_rate"])
    results["borrower_default_rates"].append(metrics["borrower_default_rate"])
    results["pnl_histories"].append(pnl_history)
    results["sector_approvals"].append(dict(sim.sector_approvals))
    results["sector_requests"].append(dict(sim.sector_requests))
    results["sector_pnls"].append(dict(sim.sector_pnl))
    # abs
    results["sharpe_ratios"].append(metrics.get("sharpe_ratio", 0.0))
    results["roas"].append(metrics.get("roa", 0.0))
    results["survival_rates"].append(metrics.get("survival_rate", 0.0))
    results["max_cascade_depths"].append(metrics.get("max_cascade_depth", 0))
    # relative
    results["pnl_per_step"].append(metrics.get("pnl_per_step", 0.0))
    results["pnl_per_company"].append(metrics.get("pnl_per_company", 0.0))


def _aggregate_results(results: dict, num_episodes: int) -> dict:
    return {
        "mean_pnl": np.mean(results["cumulative_pnls"]),
        "std_pnl": np.std(results["cumulative_pnls"]),
        "mean_defaults": np.mean(results["total_defaults"]),
        "mean_approval_rate": np.mean(results["approval_rates"]),
        "mean_borrower_default_rate": np.mean(results["borrower_default_rates"]),
        "pnl_histories": results["pnl_histories"],
        "sector_approvals": results["sector_approvals"],
        "sector_requests": results["sector_requests"],
        "sector_pnls": results["sector_pnls"],
        "all_pnls": results["cumulative_pnls"],
        "all_borrower_default_rates": results["borrower_default_rates"],
        "num_episodes": num_episodes,
        # abs
        "mean_sharpe": np.mean(results["sharpe_ratios"]),
        "mean_roa": np.mean(results["roas"]),
        "mean_survival_rate": np.mean(results["survival_rates"]),
        "mean_cascade_depth": np.mean(results["max_cascade_depths"]),
        # relative
        "mean_pnl_per_step": np.mean(results["pnl_per_step"]),
        "mean_pnl_per_company": np.mean(results["pnl_per_company"]),
    }
