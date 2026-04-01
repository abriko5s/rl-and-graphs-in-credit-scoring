"""
H2 stuff
Hypothesis: RL-agent can issue riskier loans
for promising sectors, which has a positive impact on overall PnL

Algo:
  1. Create an environment with heterogeneous sectors
  2. Reuse the model from H1 scale sweep (b-a N=50) or retrain
  3. Analyzing approval rate and P&L by sector
  4. Check the results
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from credit_scoring.config import Config
from credit_scoring.experiments.h1_graph_vs_tabular import _run_single_topology
from credit_scoring.visualization.plots import (
    plot_comparison,
    plot_percentage_profit,
    plot_sector_analysis,
)


def run_h2_experiment(
    config: Config,
    output_dir: str = "outputs/h2",
    num_eval_episodes: int = 20,
    seed: int = 42,
    skip_training: bool = False,
    model_dir: str | None = None,
) -> dict:
    """
    Запускаем H2.

    Args:
        model_dir: явный путь к директории с обученными моделями.
                   Если None — используется стандартная эвристика (h1_scale n50).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = deepcopy(config)

    # Выбираем откуда грузить модели
    if model_dir is not None:
        src_dir = Path(model_dir)
        print("=" * 60)
        print(f"H2: Reusing models from {src_dir}")
        print("=" * 60)
        results = _run_single_topology(
            cfg, str(src_dir), num_eval_episodes, seed, skip_training=True,
        )
    else:
        h1_model_dir = Path("outputs/h1_scale/barabasi_albert/n50")
        if not skip_training and _has_trained_models(h1_model_dir):
            print("=" * 60)
            print("H2: Reusing models from H1 scale sweep (BA N=50)")
            print("=" * 60)
            results = _run_single_topology(
                cfg, str(h1_model_dir), num_eval_episodes, seed, skip_training=True,
            )
        else:
            print("=" * 60)
            print("H2: Training ALL agents on heterogeneous sectors")
            print("=" * 60)
            results = _run_single_topology(
                cfg, str(out), num_eval_episodes, seed, skip_training,
            )

    avg_histories = {}
    for name, res in results.items():
        max_len = max(len(h) for h in res["pnl_histories"])
        padded = [h + [0.0] * (max_len - len(h)) for h in res["pnl_histories"]]
        avg_histories[name] = np.mean(padded, axis=0).tolist()

    plot_comparison(
        avg_histories,
        title="H2: Sector Exploration -- P&L Comparison",
        save_path=out / "h2_pnl_comparison.png",
        num_active_steps=cfg.num_steps,
    )

    plot_percentage_profit(
        results,
        title="H2: Sector Exploration -- Percentage Profit",
        save_path=out / "h2_profit.png",
    )

    print("\n" + "=" * 60)
    print("H2 RESULTS: Approval Rate by Sector (normalized by requests)")
    print("=" * 60)

    # Нормализуем одобрения на количество заявок от сектора
    sector_approval_rates = {}
    sector_pnls = {}

    for name, res in results.items():
        avg_rates = {}
        avg_pnls = {}
        for s in range(cfg.num_sectors):
            total_approvals = np.mean([ep[s] for ep in res["sector_approvals"]])
            total_requests = np.mean([ep.get(s, 1) for ep in res.get("sector_requests", [{}])])
            avg_rates[s] = total_approvals / max(total_requests, 1)
            avg_pnls[s] = np.mean([ep[s] for ep in res["sector_pnls"]])

        sector_approval_rates[name] = avg_rates
        sector_pnls[name] = avg_pnls

        print(f"\n  {name}:")
        for s in range(cfg.num_sectors):
            risk_tag = " [RISKY]" if s >= cfg.num_sectors - max(1, int(cfg.num_sectors * cfg.risky_sector_fraction)) else ""
            print(f"    Sector {s}{risk_tag}: approval_rate={avg_rates[s]:.2%}, "
                  f"P&L={avg_pnls[s]:.2f}")

    plot_sector_analysis(
        sector_approval_rates, sector_pnls,
        save_path=out / "h2_sector_analysis.png",
    )

    return results


def _has_trained_models(model_dir: Path) -> bool:
    """
    Проверить наличие обученных моделей в директории
    """
    if not model_dir.exists():
        return False
    gnn_ok = (model_dir / "gnn_gcn_ppo" / "gnn_gcn_ppo_final.zip").exists()
    mlp_ok = (model_dir / "mlp_ppo" / "mlp_ppo_final.zip").exists()
    cat_ok = (model_dir / "catboost" / "catboost_model.cbm").exists()
    return gnn_ok and mlp_ok and cat_ok
