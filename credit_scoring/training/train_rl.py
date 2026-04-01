"""
Using Stable-Baselines3 for agents training
PPO | A2C | DQN (rip)
"""

from __future__ import annotations

import sys
from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from credit_scoring.agents.gnn_agent import create_gnn_agent
from credit_scoring.agents.mlp_agent import create_mlp_agent
from credit_scoring.config import Config
from credit_scoring.env import CreditScoringEnv, FlatCreditScoringEnv
from credit_scoring.visualization.plots import MetricsCallback


def train_agent(
    agent_type: str,
    config: Config,
    output_dir: str = "outputs",
    seed: int | None = None,
    algorithm: str | None = None,
    suffix_override: str | None = None,
) -> PPO | A2C:
    """
    Обучить
    Args:
        agent_type: mlp | gnn
        config: конфигурация проекта
        output_dir: папка для сохранения моделей и логов
        seed: seed
        algorithm: ppo | a2c
        suffix_override: нейминг

    Returns:
        обученный агент
    """
    algo = algorithm or config.rl.algorithm
    if suffix_override:
        suffix = suffix_override
    elif agent_type == "gnn":
        conv = config.gnn.conv_type.lower()
        suffix = f"gnn_{conv}_{algo.lower()}"
    else:
        suffix = f"{agent_type}_{algo.lower()}"
    out = Path(output_dir) / suffix
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)

    if agent_type == "mlp":
        model = create_mlp_agent(config, seed=seed, algorithm=algo)
        eval_env = Monitor(FlatCreditScoringEnv(config=config, seed=(seed or 0) + 1000))
    elif agent_type == "gnn":
        model = create_gnn_agent(config, seed=seed, algorithm=algo)
        eval_env = Monitor(CreditScoringEnv(config=config, seed=(seed or 0) + 1000))
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Use 'mlp' or 'gnn'")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, config.rl.total_timesteps // 10),
        save_path=str(out / "checkpoints"),
        name_prefix=suffix,
    )

    eval_freq = model.n_steps * 5

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "best"),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    metrics_cb = MetricsCallback()
    cb_list = [checkpoint_cb, eval_cb, metrics_cb]
    callbacks = CallbackList(cb_list)

    # logs & other stuff
    tee = sys.stdout
    if hasattr(tee, "paused"):
        tee.paused = True
    model.learn(
        total_timesteps=config.rl.total_timesteps,
        callback=callbacks,
        tb_log_name=suffix,
    )
    if hasattr(tee, "paused"):
        tee.paused = False

    model.save(str(out / f"{suffix}_final"))
    print(f"Model saved to {out / f'{suffix}_final'}")

    return model
