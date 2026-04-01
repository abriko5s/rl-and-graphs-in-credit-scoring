"""
Stress testing of agents
Tests the resistance of agents to exogenous shocks:
    1. Bankruptcy of a high degree node
    2. Sector shock
    3. burn_rate increase
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from credit_scoring.config import Config
from credit_scoring.evaluation.evaluate import evaluate_heuristic_agent, evaluate_rl_agent


def stress_test_with_shocks(
    agent,
    config: Config,
    agent_type: str = "heuristic",
    model=None,
    num_episodes: int = 10,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """
    Прогнать агента по сценариям

    Args:
        agent: эвристика/CatBoost (None для RL)
        config: конфиг
        agent_type: heuristic | catboost | mlp | gnn
        model: SB3 (для RL)
        num_episodes: количество эпизодов на сценарий
        seed: seed

    Returns:
        {scenario_name: evaluation_results}
    """
    scenarios = {
        "baseline": config,
        "hub_shock": _config_with_hub_shock(config),
        "sector_shock": _config_with_sector_shock(config),
        "high_volatility": _config_with_high_volatility(config),
    }

    results = {}
    for name, cfg in scenarios.items():
        print(f"Running scenario: {name}")

        if agent_type in ("mlp", "gnn"):
            res = evaluate_rl_agent(
                model, cfg, num_episodes=num_episodes, seed=seed,
                use_flat_env=(agent_type == "mlp"),
            )
        else:
            res = evaluate_heuristic_agent(
                agent, cfg, num_episodes=num_episodes, seed=seed,
            )

        results[name] = res
        print(f"P&L: {res['mean_pnl']:.2f} +/- {res['std_pnl']:.2f}, "
              f"Defaults: {res['mean_defaults']:.1f}")

    return results


def _config_with_hub_shock(config: Config) -> Config:
    """
    частые индивидаульные шоки
    """
    cfg = deepcopy(config)
    cfg.shock_probability = 0.06
    cfg.shock_asset_loss = 0.5
    return cfg


def _config_with_sector_shock(config: Config) -> Config:
    """
    частые секторные шоки
    """
    cfg = deepcopy(config)
    cfg.sector_shock_probability = 0.02
    cfg.sector_shock_asset_loss = 0.25
    return cfg


def _config_with_high_volatility(config: Config) -> Config:
    """
    все шоки сильнее
    """
    cfg = deepcopy(config)
    cfg.shock_probability = 0.04
    cfg.shock_asset_loss = 0.4
    cfg.sector_shock_probability = 0.01
    cfg.sector_shock_asset_loss = 0.2
    cfg.macro_shock_probability = 0.005
    cfg.macro_shock_asset_loss = 0.12
    return cfg


def compare_stress_results(
    all_results: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """
    Выводим таблицу сравнения агентов

    Args:
        all_results: {agent_name: {scenario: eval_results}}
    """
    agents = list(all_results.keys())
    scenarios = list(next(iter(all_results.values())).keys())

    #disclaimer: vibecoded for beautiful table
    header = f"{'Agent':<20}" + "".join(f"  {s:<20}" for s in scenarios)
    print("\n" + "=" * len(header))
    print("STRESS TEST RESULTS (Mean P&L)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for agent in agents:
        row = f"{agent:<20}"
        for scenario in scenarios:
            res = all_results[agent][scenario]
            row += f"  {res['mean_pnl']:>8.1f} +/- {res['std_pnl']:<8.1f}"
        print(row)

    print("=" * len(header))
