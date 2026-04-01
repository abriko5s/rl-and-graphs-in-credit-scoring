"""
Graphs vs Tabular

Hypothesis: GNN agent reduces loan losses compared to
with MLP agent (RL without graph) and CatBoost (Classic ML).
Using:
  1. GNN x {PPO, A2C} x {GCN, GAT}
  2. MLP x {PPO, A2C}
  3. CatBoost x {static, adaptive}
  4. Heuristics: Always Approve, Always Reject, Equity Threshold
Modes:
1. Multi-topological: BA, ER, WS at a fixed size graph
2. Scale sweep: fixed topology (BA), different graph sizes
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy import stats

from credit_scoring.agents.gnn_agent import create_gnn_agent, load_gnn_agent
from credit_scoring.agents.heuristic_agent import (
    AlwaysApproveAgent,
    AlwaysRejectAgent,
    EquityThresholdAgent,
)
from credit_scoring.agents.mlp_agent import create_mlp_agent, load_mlp_agent
from credit_scoring.baselines.classical_ml import CatBoostAgent, CatBoostAdaptiveAgent
from credit_scoring.config import Config
from credit_scoring.evaluation.evaluate import evaluate_heuristic_agent, evaluate_rl_agent
from credit_scoring.training.train_classical import train_classical_baseline
from credit_scoring.training.train_rl import train_agent
from credit_scoring.visualization.plots import plot_comparison, plot_percentage_profit

# Ablation = GNN без ребер (видим только node features)
AGENT_CONFIGS = [
    ("gnn", "PPO", "GCN", False),
    ("gnn", "PPO", "GAT", False),
    ("gnn", "A2C", "GCN", False),
    ("gnn", "A2C", "GAT", False),
    ("mlp", "PPO", None, False),
    ("mlp", "A2C", None, False),
    ("gnn", "PPO", "GCN", True),  # ablation
]


def _run_single_topology(
        config: Config,
        output_dir: str,
        num_eval_episodes: int,
        seed: int,
        skip_training: bool,
) -> dict:
    """
    Запустить H1 для одной топологии графа
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = {}

    # rl
    total = len(AGENT_CONFIGS)
    for i, (agent_type, algo, conv_type, no_edges) in enumerate(AGENT_CONFIGS, 1):
        # naming stuff
        if agent_type == "gnn" and conv_type:
            if no_edges:
                label = f"GNN-{conv_type} ({algo}, no graph)"
                suffix = f"gnn_{conv_type.lower()}_{algo.lower()}_no_graph"
            else:
                label = f"GNN-{conv_type} ({algo})"
                suffix = f"gnn_{conv_type.lower()}_{algo.lower()}"
        else:
            label = f"MLP ({algo})"
            suffix = f"mlp_{algo.lower()}"

        # conv_type and ablation
        run_config = deepcopy(config)
        if conv_type:
            run_config.gnn.conv_type = conv_type
        if no_edges:
            run_config.ablation_no_edges = True

        if not skip_training:
            print(f"\n[{i}/{total}] Training {label} ({config.graph_type})")
            model = train_agent(
                agent_type, run_config, output_dir=str(out), seed=seed,
                algorithm=algo, suffix_override=suffix,
            )
        else:
            model_path = out / suffix / f"{suffix}_final"
            if agent_type == "gnn":
                model = load_gnn_agent(model_path, run_config, seed=seed, algorithm=algo)
            else:
                model = load_mlp_agent(model_path, run_config, seed=seed, algorithm=algo)

        print(f"  Evaluating {label}")
        use_flat = (agent_type == "mlp")
        results[label] = evaluate_rl_agent(
            model, run_config, num_episodes=num_eval_episodes,
            seed=seed + 5000, use_flat_env=use_flat,
        )

    # catboost
    if not skip_training:
        print(f"\nTraining CatBoost ({config.graph_type})")
        catboost_agent = train_classical_baseline(config, output_dir=str(out), seed=seed)
    else:
        catboost_agent = CatBoostAgent.load(
            out / "catboost" / "catboost_model.cbm",
            threshold=config.catboost.approval_threshold,
        )

    print("Evaluating CatBoost (static)")
    results["CatBoost (static)"] = evaluate_heuristic_agent(
        catboost_agent, config, num_episodes=num_eval_episodes, seed=seed + 5000,
    )

    # CatBoost Adaptive: тот же классификатор, но адаптивный threshold
    adaptive_agent = CatBoostAdaptiveAgent(
        catboost_agent.model,
        base_threshold=config.catboost.approval_threshold,
    )
    print("  Evaluating CatBoost (adaptive)...")
    results["CatBoost (adaptive)"] = evaluate_heuristic_agent(
        adaptive_agent, config, num_episodes=num_eval_episodes, seed=seed + 5000,
    )

    # heuristics
    for name, agent in [
        ("Always Approve", AlwaysApproveAgent()),
        ("Equity Threshold", EquityThresholdAgent(threshold=50.0)),
        ("Always Reject", AlwaysRejectAgent()),
    ]:
        print(f"Evaluating {name}")
        results[name] = evaluate_heuristic_agent(
            agent, config, num_episodes=num_eval_episodes, seed=seed + 5000,
        )

    return results


def run_h1_experiment(
        config: Config,
        output_dir: str = "outputs/h1",
        num_eval_episodes: int = 20,
        seed: int = 42,
        skip_training: bool = False,
) -> dict:
    """
    Запустить полный эксперимент H1 на нескольких топологиях графов
    Returns:
        словарь {topology_name: {agent_name: results}}
    """
    topologies = [
        ("barabasi_albert", "BA (scale-free)"),
        ("erdos_renyi", "ER (random)"),
        ("watts_strogatz", "WS (small-world)"),
    ]

    all_results = {}

    for graph_type, label in topologies:
        print("\n" + "=" * 60)
        print(f"H1: Topology = {label}")
        print("=" * 60)

        topo_config = deepcopy(config)
        topo_config.graph_type = graph_type

        topo_dir = str(Path(output_dir) / graph_type)
        results = _run_single_topology(
            topo_config, topo_dir, num_eval_episodes, seed, skip_training,
        )
        all_results[label] = results

    # results
    _print_summary(all_results)

    # graphics
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for topo_label, results in all_results.items():
        avg_histories = {}
        for name, res in results.items():
            max_len = max(len(h) for h in res["pnl_histories"])
            padded = [h + [0.0] * (max_len - len(h)) for h in res["pnl_histories"]]
            avg_histories[name] = np.mean(padded, axis=0).tolist()

        safe_name = topo_label.replace(" ", "_").replace("(", "").replace(")", "")
        plot_comparison(
            avg_histories,
            title=f"H1: {topo_label} — P&L Comparison",
            save_path=out / f"h1_{safe_name}.png",
            num_active_steps=config.num_steps,
        )

        # ROA %
        plot_percentage_profit(
            results,
            title=f"H1: {topo_label} — Percentage Profit",
            save_path=out / f"h1_{safe_name}_profit.png",
        )

    return all_results


def _print_summary(all_results: dict) -> None:
    """
    Напечатать сводную табличку
    """
    print("\n" + "=" * 100)
    print("H1 SUMMARY: ALL TOPOLOGIES × ALL ALGORITHMS")
    print("=" * 100)

    # vibecoded для ровной таблички
    for topo_label, results in all_results.items():
        print(f"\n--- {topo_label} ---")
        print(f"  {'Agent':<28} {'P&L':>10} {'±std':>8} {'P&L/step':>9} {'P&L/co':>8} "
              f"{'Defaults':>9} {'Appr%':>7} {'BorrDR':>8} {'Sharpe':>8} {'ROA':>8} "
              f"{'Surv%':>7} {'Casc':>5}")
        print(f"  {'-' * 123}")

        # pnl sort
        sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_pnl"], reverse=True)
        for name, res in sorted_results:
            bdr = res.get('mean_borrower_default_rate', 0)
            sharpe = res.get('mean_sharpe', 0)
            roa = res.get('mean_roa', 0)
            surv = res.get('mean_survival_rate', 0)
            casc = res.get('mean_cascade_depth', 0)
            pps = res.get('mean_pnl_per_step', 0)
            ppc = res.get('mean_pnl_per_company', 0)
            print(
                f"  {name:<28} {res['mean_pnl']:>10.1f} {res['std_pnl']:>7.1f} "
                f"{pps:>9.3f} {ppc:>8.2f} "
                f"{res['mean_defaults']:>9.1f} {res['mean_approval_rate']:>6.1%} "
                f"{bdr:>7.1%} {sharpe:>8.3f} {roa:>8.3f} {surv:>6.1%} {casc:>5.1f}"
            )

    # gnn vs mlp
    print(f"\n{'=' * 100}")
    print("GNN vs MLP COMPARISON (Welch t-test):")
    print(f"  {'Comparison':<35} {'Topology':<23} {'Left P&L':>10} {'Right P&L':>10} {'Delta':>10} {'p-value':>10}")
    print(f"  {'-' * 100}")

    for algo in ["PPO", "A2C"]:
        for conv in ["GCN", "GAT"]:
            gnn_label = f"GNN-{conv} ({algo})"
            mlp_label = f"MLP ({algo})"

            for topo_label, results in all_results.items():
                if gnn_label in results and mlp_label in results:
                    gnn_pnls = results[gnn_label]["all_pnls"]
                    mlp_pnls = results[mlp_label]["all_pnls"]
                    gnn_mean = np.mean(gnn_pnls)
                    mlp_mean = np.mean(mlp_pnls)
                    delta = gnn_mean - mlp_mean
                    _, p_val = stats.ttest_ind(gnn_pnls, mlp_pnls, equal_var=False)

                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                    comp = f"{gnn_label} vs {mlp_label}"
                    print(
                        f"  {comp:<35} {topo_label:<23} {gnn_mean:>10.1f} {mlp_mean:>10.1f} "
                        f"{delta:>+10.1f} {p_val:>9.4f} {sig}"
                    )

    # gcn vs gat
    print(f"\n  {'GCN vs GAT (same algorithm):':}")
    print(f"  {'-' * 100}")
    for algo in ["PPO", "A2C"]:
        gcn_label = f"GNN-GCN ({algo})"
        gat_label = f"GNN-GAT ({algo})"

        for topo_label, results in all_results.items():
            if gcn_label in results and gat_label in results:
                gcn_pnls = results[gcn_label]["all_pnls"]
                gat_pnls = results[gat_label]["all_pnls"]
                gcn_mean = np.mean(gcn_pnls)
                gat_mean = np.mean(gat_pnls)
                delta = gat_mean - gcn_mean
                _, p_val = stats.ttest_ind(gcn_pnls, gat_pnls, equal_var=False)

                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                comp = f"{gat_label} vs {gcn_label}"
                print(
                    f"  {comp:<35} {topo_label:<23} {gat_mean:>10.1f} {gcn_mean:>10.1f} "
                    f"{delta:>+10.1f} {p_val:>9.4f} {sig}"
                )

    # ablation
    ablation_label = "GNN-GCN (PPO, no graph)"
    gnn_full_label = "GNN-GCN (PPO)"
    has_ablation = any(ablation_label in results for results in all_results.values())
    if has_ablation:
        print(f"\n  {'ABLATION: graph structure contribution':}")
        print(f"  {'-' * 100}")
        for topo_label, results in all_results.items():
            if gnn_full_label in results and ablation_label in results:
                full_pnls = results[gnn_full_label]["all_pnls"]
                noedge_pnls = results[ablation_label]["all_pnls"]
                full_mean = np.mean(full_pnls)
                noedge_mean = np.mean(noedge_pnls)
                delta = full_mean - noedge_mean
                _, p_val = stats.ttest_ind(full_pnls, noedge_pnls, equal_var=False)

                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                comp = f"{gnn_full_label} vs {ablation_label}"
                print(
                    f"  {comp:<45} {topo_label:<15} {full_mean:>10.1f} {noedge_mean:>10.1f} "
                    f"{delta:>+10.1f} {p_val:>9.4f} {sig}"
                )
    # красиво подвайбкоженые звездочки со значимостью
    print(f"\n  Significance: *** p<0.01, ** p<0.05, * p<0.1")


# Scale sweep

_BASE_COMPANIES = 50
_BASE_STEPS = 600
_TIMESTEPS_BY_SIZE = {
    5: 50_000,
    10: 150_000,
    20: 300_000,
    50: 500_000,
    100: 1_000_000,
}


def _scale_config(config: Config, num_companies: int) -> Config:
    """
    Адаптировать конфиг под другой размер графа
        - num_steps
        - total_timesteps
        - ba_edges_per_node: min(3, num_companies - 1)
        - er_prob
        - ws_k: min(4, num_companies - 1)
    """
    scaled = deepcopy(config)

    scaled.num_companies = num_companies
    scaled.num_steps = max(50, int(_BASE_STEPS * num_companies / _BASE_COMPANIES))
    scaled.rl.total_timesteps = _TIMESTEPS_BY_SIZE.get(
        num_companies, max(30_000, int(300_000 * num_companies / _BASE_COMPANIES))
    )

    # хотим среднюю степень 6
    scaled.ba_edges_per_node = min(3, num_companies - 1)
    scaled.er_prob = min(0.9, 6.0 / max(1, num_companies - 1))  # avg degree ≈ 6
    scaled.ws_k = min(4, num_companies - 1)
    if scaled.ws_k % 2 != 0:
        scaled.ws_k = max(2, scaled.ws_k - 1)  # ws_k must be even

    return scaled


def run_h1_scale_experiment(
        config: Config,
        output_dir: str = "outputs/h1_scale",
        graph_sizes: list[int] | None = None,
        num_eval_episodes: int = 20,
        seed: int = 42,
        skip_training: bool = False,
) -> dict:
    """
    Запустить H1 на разных размерах графа x topologies
    Показывает, как масштабируется преимущество GNN с ростом графа
    на разных типах графовых структур
    Args:
        config: базовый конфиг
        output_dir: папка для результатов
        graph_sizes: [5, 10, 20, 50]
        num_eval_episodes: эпизоды для оценки
        seed: seed
        skip_training: просто посчитать

    Returns:
        словарь {label: {agent_name: results}}
    """
    if graph_sizes is None:
        graph_sizes = [5, 10, 20, 50]

    topologies = [
        ("barabasi_albert", "BA"),
        ("erdos_renyi", "ER"),
        ("watts_strogatz", "WS"),
    ]

    all_results = {}

    for graph_type, topo_short in topologies:
        for n in graph_sizes:
            label = f"{topo_short} N={n}"
            print("\n" + "=" * 60)
            print(f"H1 SCALE: {label}")
            print("=" * 60)

            scaled_config = _scale_config(config, n)
            scaled_config.graph_type = graph_type

            print(f"  num_companies={n}, num_steps={scaled_config.num_steps}, "
                  f"total_timesteps={scaled_config.rl.total_timesteps}")

            size_dir = str(Path(output_dir) / graph_type / f"n{n}")
            results = _run_single_topology(
                scaled_config, size_dir, num_eval_episodes, seed, skip_training,
            )
            all_results[label] = results

    _print_summary(all_results)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for label, results in all_results.items():
        avg_histories = {}
        for name, res in results.items():
            max_len = max(len(h) for h in res["pnl_histories"])
            padded = [h + [0.0] * (max_len - len(h)) for h in res["pnl_histories"]]
            avg_histories[name] = np.mean(padded, axis=0).tolist()

        # yet another parsing
        parts = label.split()
        topo_short = parts[0]
        n_int = int(parts[1].replace("N=", ""))
        scaled_steps = max(50, int(_BASE_STEPS * n_int / _BASE_COMPANIES))

        safe_label = label.replace(" ", "_").replace("=", "")
        plot_comparison(
            avg_histories,
            title=f"H1 Scale: {label} — P&L Comparison",
            save_path=out / f"h1_scale_{safe_label}.png",
            num_active_steps=scaled_steps,
        )

        plot_percentage_profit(
            results,
            title=f"H1 Scale: {label} — Percentage Profit",
            save_path=out / f"h1_scale_{safe_label}_profit.png",
        )

    # gnn adv на каждом размере
    for _, topo_short in topologies:
        topo_results = {k: v for k, v in all_results.items() if k.startswith(topo_short)}
        if topo_results:
            print(f"\n  --- {topo_short} ---")
            _print_scale_summary(topo_results)

    return all_results


def _print_scale_summary(all_results: dict) -> None:
    """
    Напечатать сводку GNN advantage по размерам графа.
    """
    print(f"\n{'=' * 100}")
    print("SCALE ANALYSIS: GNN advantage vs graph size")
    print(f"{'=' * 100}")
    print(f"  {'Size':<8} {'Best GNN':>12} {'Best MLP':>12} {'Delta':>10} "
          f"{'GNN BorrDR':>10} {'MLP BorrDR':>10} {'GNN Surv%':>10} {'MLP Surv%':>10}")
    print(f"  {'-' * 90}")

    for size_label, results in all_results.items():
        gnn_results = {k: v for k, v in results.items() if k.startswith("GNN") and "no graph" not in k}
        mlp_results = {k: v for k, v in results.items() if k.startswith("MLP")}

        if not gnn_results or not mlp_results:
            continue

        best_gnn_name = max(gnn_results, key=lambda k: gnn_results[k]["mean_pnl"])
        best_mlp_name = max(mlp_results, key=lambda k: mlp_results[k]["mean_pnl"])

        gnn = gnn_results[best_gnn_name]
        mlp = mlp_results[best_mlp_name]

        delta = gnn["mean_pnl"] - mlp["mean_pnl"]
        # vibecoded
        print(
            f"  {size_label:<8} {gnn['mean_pnl']:>10.1f}   {mlp['mean_pnl']:>10.1f}   "
            f"{delta:>+10.1f} {gnn.get('mean_borrower_default_rate', 0):>9.1%} "
            f"{mlp.get('mean_borrower_default_rate', 0):>10.1%} "
            f"{gnn.get('mean_survival_rate', 0):>9.1%} "
            f"{mlp.get('mean_survival_rate', 0):>10.1%}"
        )
