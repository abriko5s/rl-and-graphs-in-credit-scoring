"""
CatBoost training
Collecting data from environment, learning classifier and saving model
"""

from __future__ import annotations

import sys
from pathlib import Path

from credit_scoring.baselines.classical_ml import (
    CatBoostAgent,
    collect_dataset,
    train_catboost,
)
from credit_scoring.config import Config


def train_classical_baseline(
        config: Config,
        output_dir: str = "outputs",
        num_episodes: int = 100,
        seed: int = 42,
) -> CatBoostAgent:
    """
    Обучить CatBoost-бейзлайн

    Args:
        config: конфигурация
        output_dir: папка для сохранения
        num_episodes: количество эпизодов для сбора данных
        seed: seed

    Returns:
        обученный CatBoostAgent
    """
    out = Path(output_dir) / "catboost"
    out.mkdir(parents=True, exist_ok=True)

    print(f"Collecting dataset ({num_episodes} episodes)")
    X, y = collect_dataset(config, num_episodes=num_episodes, seed=seed)
    print(f"Dataset: {X.shape[0]} samples, {y.mean():.1%} defaults")

    print("Training CatBoost")
    tee = sys.stdout
    if hasattr(tee, "paused"):
        tee.paused = True
    model = train_catboost(X, y, config)
    if hasattr(tee, "paused"):
        tee.paused = False

    agent = CatBoostAgent(model, threshold=config.catboost.approval_threshold)
    agent.save(out / "catboost_model.cbm")
    print(f"Model saved to {out / 'catboost_model.cbm'}")

    return agent
