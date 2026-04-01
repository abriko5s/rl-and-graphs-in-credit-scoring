"""
MLP-agent
Using only table features of company and application
(PPO | A2C)
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import A2C, PPO

from credit_scoring.config import Config
from credit_scoring.env import FlatCreditScoringEnv


def create_mlp_agent(
        config: Config, seed: int | None = None, algorithm: str | None = None,
) -> PPO | A2C:
    """
    Создать агента
    Args:
        config: конфиг
        seed: очев
        algorithm: PPO | A2C (None = из конфига)
    Returns:
        model: инициализиованный агент
    """
    env = FlatCreditScoringEnv(config=config, seed=seed)
    rl = config.rl
    actual_seed = rl.seed if seed is None else seed
    algo = algorithm or rl.algorithm

    if algo == "A2C":
        return A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=rl.learning_rate,
            n_steps=min(rl.n_steps, 64),
            gamma=rl.gamma,
            gae_lambda=rl.gae_lambda,
            ent_coef=rl.ent_coef,
            seed=actual_seed,
            verbose=1,
        )
    else:  # TODO: убрать для масштабируемости на другие алгосы
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=rl.learning_rate,
            n_steps=rl.n_steps,
            batch_size=rl.batch_size,
            gamma=rl.gamma,
            gae_lambda=rl.gae_lambda,
            ent_coef=rl.ent_coef,
            n_epochs=rl.n_epochs,
            seed=actual_seed,
            verbose=1,
        )


def load_mlp_agent(
        path: str | Path, config: Config, seed: int | None = None,
        algorithm: str | None = None,
) -> PPO | A2C:
    """
    Загрузить агента из файла
    """
    env = FlatCreditScoringEnv(config=config, seed=seed)
    algo = algorithm or config.rl.algorithm

    cls = A2C if algo == "A2C" else PPO  # TODO: аккуратно если будут новые алгосы
    return cls.load(str(path), env=env)
