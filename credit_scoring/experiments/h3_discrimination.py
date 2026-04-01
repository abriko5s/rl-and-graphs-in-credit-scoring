"""
H3 stuff

Hypothesis: gnn may unreasonably downgrade the ratings of trustworthy companies
because of their position in the graph
Algo:
  1. Train gnn and mlp agents
  2. Recording
     -node degree
     -approve rate
  3. Consider correlation (pearson / spearman)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from credit_scoring.agents.gnn_agent import create_gnn_agent, load_gnn_agent
from credit_scoring.agents.mlp_agent import create_mlp_agent, load_mlp_agent
from credit_scoring.config import Config
from credit_scoring.env import CreditScoringEnv, FlatCreditScoringEnv
from credit_scoring.training.train_rl import train_agent
from credit_scoring.visualization.plots import plot_degree_vs_approval


def run_h3_experiment(
        config: Config,
        output_dir: str = "outputs/h3",
        num_eval_episodes: int = 30,
        seed: int = 42,
        skip_training: bool = False,
) -> dict:
    """
    Запустить H3
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    h1_scale_dir = Path("outputs/h1_scale/barabasi_albert/n50")
    h1_ba_dir = Path("outputs/h1/barabasi_albert")

    gnn_model = None
    mlp_model = None

    if not skip_training:
        for search_dir in [h1_scale_dir, h1_ba_dir]:
            gnn_path = search_dir / "gnn_gcn_ppo" / "gnn_gcn_ppo_final.zip"
            mlp_path = search_dir / "mlp_ppo" / "mlp_ppo_final.zip"
            if gnn_path.exists() and mlp_path.exists():
                print(f"Reusing models from {search_dir}...")
                gnn_model = load_gnn_agent(
                    gnn_path.with_suffix(""), config, seed=seed, algorithm="PPO",
                )
                mlp_model = load_mlp_agent(
                    mlp_path.with_suffix(""), config, seed=seed, algorithm="PPO",
                )
                break
        else:
            print("=" * 60)
            print("H3: Training GNN and MLP agents")
            print("=" * 60)
            gnn_model = train_agent(
                "gnn", config, output_dir=str(out), seed=seed,
                algorithm="PPO",
            )
            mlp_model = train_agent(
                "mlp", config, output_dir=str(out), seed=seed,
                algorithm="PPO",
            )
    else:
        conv = config.gnn.conv_type.lower()
        gnn_model = load_gnn_agent(
            out / f"gnn_{conv}_ppo" / f"gnn_{conv}_ppo_final", config, seed=seed, algorithm="PPO",
        )
        mlp_model = load_mlp_agent(
            out / "mlp_ppo" / "mlp_ppo_final", config, seed=seed, algorithm="PPO",
        )

    print("\nCollecting approval decisions per company")

    gnn_data = _collect_degree_approval_data(
        gnn_model, config, num_eval_episodes, seed + 5000, use_flat=False,
    )
    mlp_data = _collect_degree_approval_data(
        mlp_model, config, num_eval_episodes, seed + 5000, use_flat=True,
    )

    print("\n" + "=" * 60)
    print("H3 RESULTS: Structural Discrimination Analysis")
    print("=" * 60)

    results = {}

    for name, data in [("GNN-GCN (PPO)", gnn_data), ("MLP (PPO)", mlp_data)]:
        degrees = data["degrees"]
        approval_rates = data["approval_rates"]

        if len(degrees) < 3:
            print(f"  {name}: insufficient data")
            continue

        pearson_r, pearson_p = stats.pearsonr(degrees, approval_rates)
        spearman_r, spearman_p = stats.spearmanr(degrees, approval_rates)

        results[name] = {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "degrees": degrees,
            "approval_rates": approval_rates,
        }

        # vibecoded prints
        print(f"\n  {name}:")
        print(f"    Pearson:  r={pearson_r:.4f}, p={pearson_p:.4f}")
        print(f"    Spearman: r={spearman_r:.4f}, p={spearman_p:.4f}")

        if pearson_p < 0.05:
            direction = "positive" if pearson_r > 0 else "negative"
            print(f"    => Significant {direction} correlation (p < 0.05)")
            if pearson_r > 0:
                print("    => Companies with MORE connections get approved more often")
                print("    => POSSIBLE structural discrimination against low-degree nodes")
            else:
                print("    => Companies with FEWER connections get approved more often")
        else:
            print("    => No significant correlation (p >= 0.05)")
            print("    => No evidence of structural discrimination")

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_degree_vs_approval(
            degrees, approval_rates, agent_name=name,
            save_path=out / f"h3_{safe_name}_degree_approval.png",
        )

    return results


def _collect_degree_approval_data(
        model,
        config: Config,
        num_episodes: int,
        seed: int,
        use_flat: bool,
) -> dict:
    """
    Собираем данные
    """
    degrees = []
    approval_rates = []

    for ep in range(num_episodes):
        ep_seed = seed + ep
        if use_flat:
            env = FlatCreditScoringEnv(config=config, seed=ep_seed)
        else:
            env = CreditScoringEnv(config=config, seed=ep_seed)

        obs, _ = env.reset(seed=ep_seed)

        sim = env.sim if hasattr(env, "sim") else env._full_env.sim
        G = sim.graph
        ep_degrees = {node: G.degree(node) for node in G.nodes()}

        ep_approvals = defaultdict(int)
        ep_total = defaultdict(int)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            app_id = sim.current_application.applicant_id
            ep_total[app_id] += 1
            if action == 1:
                ep_approvals[app_id] += 1

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        for cid in ep_total:
            if ep_total[cid] >= 2:
                degrees.append(ep_degrees.get(cid, 0))
                approval_rates.append(ep_approvals[cid] / ep_total[cid])

    return {"degrees": degrees, "approval_rates": approval_rates}
