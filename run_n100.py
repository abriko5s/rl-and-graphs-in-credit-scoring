"""
How to use:
  python run_n100.py
  python run_n100.py --skip-training
  python run_n100.py --topologies BA
  python run_n100.py --topologies BA WS
  python run_n100.py --seed 123

"""

from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
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

_BASE_DIR = Path(__file__).parent
CONFIG_PATH = _BASE_DIR / "configs" / "n100.yaml"
OUTPUT_DIR = _BASE_DIR / "outputs" / "n100"

NUM_EVAL_EPISODES = 20

AGENT_CONFIGS = [
    ("gnn", "PPO", "GCN", False),
    ("gnn", "PPO", "GAT", False),
    ("gnn", "A2C", "GCN", False),
    ("gnn", "A2C", "GAT", False),
    ("mlp", "PPO", None, False),
    ("mlp", "A2C", None, False),
    ("gnn", "PPO", "GCN", True),  # ablation
]

TOPOLOGY_MAP = {
    "BA": "barabasi_albert",
    "ER": "erdos_renyi",
    "WS": "watts_strogatz",
}


class TeeWriter:
    def __init__(self, filepath: Path):
        self.terminal = sys.__stdout__
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(filepath, "w", encoding="utf-8")
        self.paused = False

    def write(self, message: str) -> int:
        self.terminal.write(message)
        if not self.paused:
            self.log.write(message)
            self.log.flush()
        return len(message)

    def flush(self) -> None:
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self) -> None:
        self.log.close()
        sys.stdout = self.terminal


def build_config(graph_type: str) -> Config:
    cfg = Config.from_yaml(CONFIG_PATH)
    cfg.graph_type = graph_type
    return cfg


def _agent_label_and_suffix(
        agent_type: str, algo: str, conv_type: str | None, no_edges: bool,
) -> tuple[str, str]:
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
    return label, suffix


def run_topology(
        cfg: Config,
        topo_dir: Path,
        seed: int,
        skip_training: bool,
) -> dict[str, dict]:
    topo_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    # rl
    total = len(AGENT_CONFIGS)
    for i, (agent_type, algo, conv_type, no_edges) in enumerate(AGENT_CONFIGS, 1):
        label, suffix = _agent_label_and_suffix(agent_type, algo, conv_type, no_edges)

        run_cfg = deepcopy(cfg)
        if conv_type:
            run_cfg.gnn.conv_type = conv_type
        if no_edges:
            run_cfg.ablation_no_edges = True

        model_path = topo_dir / suffix / f"{suffix}_final"

        if not skip_training:
            print(f"\n[{i}/{total}] Training {label} ({cfg.graph_type})...")
            model = train_agent(
                agent_type,
                run_cfg,
                output_dir=str(topo_dir),
                seed=seed,
                algorithm=algo,
                suffix_override=suffix,
            )
        else:
            print(f"\n[{i}/{total}] Loading {label} from {model_path}...")
            if agent_type == "gnn":
                model = load_gnn_agent(model_path, run_cfg, seed=seed, algorithm=algo)
            else:
                model = load_mlp_agent(model_path, run_cfg, seed=seed, algorithm=algo)

        print(f"  Evaluating {label}...")
        use_flat = (agent_type == "mlp")
        results[label] = evaluate_rl_agent(
            model,
            run_cfg,
            num_episodes=NUM_EVAL_EPISODES,
            seed=seed + 5000,
            use_flat_env=use_flat,
        )

    # catboost
    catboost_path = topo_dir / "catboost" / "catboost_model.cbm"

    if not skip_training:
        print(f"\nTraining CatBoost ({cfg.graph_type})...")
        catboost_agent = train_classical_baseline(
            cfg, output_dir=str(topo_dir), seed=seed,
        )
    else:
        print(f"\nLoading CatBoost from {catboost_path}...")
        catboost_agent = CatBoostAgent.load(
            catboost_path,
            threshold=cfg.catboost.approval_threshold,
        )

    print("  Evaluating CatBoost (static)...")
    results["CatBoost (static)"] = evaluate_heuristic_agent(
        catboost_agent, cfg, num_episodes=NUM_EVAL_EPISODES, seed=seed + 5000,
    )

    adaptive_agent = CatBoostAdaptiveAgent(
        catboost_agent.model,
        base_threshold=cfg.catboost.approval_threshold,
    )
    print("  Evaluating CatBoost (adaptive)...")
    results["CatBoost (adaptive)"] = evaluate_heuristic_agent(
        adaptive_agent, cfg, num_episodes=NUM_EVAL_EPISODES, seed=seed + 5000,
    )

    # heuristics
    for name, agent in [
        ("Always Approve", AlwaysApproveAgent()),
        ("Equity Threshold", EquityThresholdAgent(threshold=50.0)),
        ("Always Reject", AlwaysRejectAgent()),
    ]:
        print(f"  Evaluating {name}...")
        results[name] = evaluate_heuristic_agent(
            agent, cfg, num_episodes=NUM_EVAL_EPISODES, seed=seed + 5000,
        )

    return results


def print_topology_table(label: str, results: dict) -> None:
    """
    Табличка для 1 вида графа
    """
    print(f"\n--- {label} ---")
    header = (
        f"  {'Agent':<28} {'P&L':>10} {'±std':>8} {'P&L/step':>9} {'P&L/co':>8} "
        f"{'Defaults':>9} {'Appr%':>7} {'BorrDR':>8} {'Sharpe':>8} {'ROA':>8} "
        f"{'Surv%':>7} {'Casc':>5}"
    )
    print(header)
    print(f"  {'-' * 123}")

    for name, res in sorted(results.items(), key=lambda x: x[1]["mean_pnl"], reverse=True):
        print(
            f"  {name:<28} {res['mean_pnl']:>10.1f} {res['std_pnl']:>7.1f} "
            f"{res.get('mean_pnl_per_step', 0):>9.3f} "
            f"{res.get('mean_pnl_per_company', 0):>8.2f} "
            f"{res['mean_defaults']:>9.1f} "
            f"{res['mean_approval_rate']:>6.1%} "
            f"{res.get('mean_borrower_default_rate', 0):>7.1%} "
            f"{res.get('mean_sharpe', 0):>8.3f} "
            f"{res.get('mean_roa', 0):>8.3f} "
            f"{res.get('mean_survival_rate', 0):>6.1%} "
            f"{res.get('mean_cascade_depth', 0):>5.1f}"
        )


def print_all_tables(all_results: dict) -> None:
    """Сводная таблица по всем топологиям."""
    print("\n" + "=" * 100)
    print("N=100 SUMMARY: ALL TOPOLOGIES × ALL ALGORITHMS")
    print("=" * 100)
    for topo_label, results in all_results.items():
        print_topology_table(topo_label, results)


def print_statistical_tests(all_results: dict) -> None:
    """Welch t-test: GNN vs MLP + аблация."""
    print(f"\n{'=' * 100}")
    print("GNN vs MLP COMPARISON (Welch t-test) — N=100:")
    print(
        f"  {'Comparison':<40} {'Topology':<12} "
        f"{'Left P&L':>10} {'Right P&L':>10} {'Delta':>10} {'p-value':>12}"
    )
    print(f"  {'-' * 100}")

    for algo in ["PPO", "A2C"]:
        for conv in ["GCN", "GAT"]:
            gnn_lbl = f"GNN-{conv} ({algo})"
            mlp_lbl = f"MLP ({algo})"
            for topo_label, results in all_results.items():
                if gnn_lbl not in results or mlp_lbl not in results:
                    continue
                gnn_pnls = results[gnn_lbl]["all_pnls"]
                mlp_pnls = results[mlp_lbl]["all_pnls"]
                delta = np.mean(gnn_pnls) - np.mean(mlp_pnls)
                _, p_val = stats.ttest_ind(gnn_pnls, mlp_pnls, equal_var=False)
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                comp = f"{gnn_lbl} vs {mlp_lbl}"
                print(
                    f"  {comp:<40} {topo_label:<12} "
                    f"{np.mean(gnn_pnls):>10.1f} {np.mean(mlp_pnls):>10.1f} "
                    f"{delta:>+10.1f} {p_val:>10.4f} {sig}"
                )

    # GAT vs GCN
    print(f"\n  GCN vs GAT:")
    print(f"  {'-' * 100}")
    for algo in ["PPO", "A2C"]:
        gcn_lbl = f"GNN-GCN ({algo})"
        gat_lbl = f"GNN-GAT ({algo})"
        for topo_label, results in all_results.items():
            if gcn_lbl not in results or gat_lbl not in results:
                continue
            gcn_pnls = results[gcn_lbl]["all_pnls"]
            gat_pnls = results[gat_lbl]["all_pnls"]
            delta = np.mean(gat_pnls) - np.mean(gcn_pnls)
            _, p_val = stats.ttest_ind(gat_pnls, gcn_pnls, equal_var=False)
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            comp = f"{gat_lbl} vs {gcn_lbl}"
            print(
                f"  {comp:<40} {topo_label:<12} "
                f"{np.mean(gat_pnls):>10.1f} {np.mean(gcn_pnls):>10.1f} "
                f"{delta:>+10.1f} {p_val:>10.4f} {sig}"
            )

    print(f"\n  ABLATION: GNN-GCN (PPO) с графом vs без:")
    print(f"  {'-' * 100}")
    full_lbl = "GNN-GCN (PPO)"
    noedge_lbl = "GNN-GCN (PPO, no graph)"
    for topo_label, results in all_results.items():
        if full_lbl not in results or noedge_lbl not in results:
            continue
        full_pnls = results[full_lbl]["all_pnls"]
        noedge_pnls = results[noedge_lbl]["all_pnls"]
        delta = np.mean(full_pnls) - np.mean(noedge_pnls)
        _, p_val = stats.ttest_ind(full_pnls, noedge_pnls, equal_var=False)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
        comp = f"{full_lbl} vs {noedge_lbl}"
        print(
            f"  {comp:<40} {topo_label:<12} "
            f"{np.mean(full_pnls):>10.1f} {np.mean(noedge_pnls):>10.1f} "
            f"{delta:>+10.1f} {p_val:>10.4f} {sig}"
        )

    print(f"\n  Significance: *** p<0.01, ** p<0.05, * p<0.1")


def print_gnn_advantage_summary(all_results: dict) -> None:
    """gnn vs mlp"""
    print(f"\n{'=' * 100}")
    print("GNN ADVANTAGE SUMMARY (N=100 vs N=50 comparison reference):")
    print(
        f"  {'Topology':<12} {'Best GNN':>20} {'P&L':>10} "
        f"{'Best MLP':>12} {'P&L':>10} {'Delta':>10} {'Surv% GNN':>10} {'Surv% MLP':>10}"
    )
    print(f"  {'-' * 100}")

    for topo_label, results in all_results.items():
        gnn_res = {
            k: v for k, v in results.items()
            if k.startswith("GNN") and "no graph" not in k
        }
        mlp_res = {k: v for k, v in results.items() if k.startswith("MLP")}
        if not gnn_res or not mlp_res:
            continue

        best_gnn = max(gnn_res, key=lambda k: gnn_res[k]["mean_pnl"])
        best_mlp = max(mlp_res, key=lambda k: mlp_res[k]["mean_pnl"])

        gnn = gnn_res[best_gnn]
        mlp = mlp_res[best_mlp]
        delta = gnn["mean_pnl"] - mlp["mean_pnl"]

        print(
            f"  {topo_label:<12} {best_gnn:>20} {gnn['mean_pnl']:>10.1f} "
            f"{best_mlp:>12} {mlp['mean_pnl']:>10.1f} {delta:>+10.1f} "
            f"{gnn.get('mean_survival_rate', 0):>9.1%} "
            f"{mlp.get('mean_survival_rate', 0):>9.1%}"
        )


def save_plots(topo_short: str, results: dict, num_steps: int) -> None:
    """Сохранить P&L-кривые и profit-chart для одной топологии."""
    avg_histories: dict[str, list[float]] = {}
    for name, res in results.items():
        max_len = max(len(h) for h in res["pnl_histories"])
        padded = [h + [0.0] * (max_len - len(h)) for h in res["pnl_histories"]]
        avg_histories[name] = np.mean(padded, axis=0).tolist()

    plot_comparison(
        avg_histories,
        title=f"N=100 {topo_short}: Cumulative P&L",
        save_path=OUTPUT_DIR / f"n100_{topo_short}_pnl.png",
        num_active_steps=num_steps,
    )
    plot_percentage_profit(
        results,
        title=f"N=100 {topo_short}: Percentage Profit",
        save_path=OUTPUT_DIR / f"n100_{topo_short}_profit.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone N=100 experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Пропустить обучение, загрузить существующие модели из outputs/n100/",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        choices=["BA", "ER", "WS"],
        default=["BA", "ER", "WS"],
        metavar="TOPO",
        help="Топологии для запуска (default: BA ER WS)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report_n100.txt"

    tee = TeeWriter(report_path)
    sys.stdout = tee

    start = time.time()
    print(f"N=100 Experiment started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Config:     {CONFIG_PATH}")
    print(f"Output:     {OUTPUT_DIR.resolve()}")
    print(f"Topologies: {args.topologies}")
    print(f"Seed:       {args.seed}")
    print(f"Skip train: {args.skip_training}")
    print()

    _probe_cfg = Config.from_yaml(CONFIG_PATH)
    num_steps = _probe_cfg.num_steps

    all_results: dict[str, dict] = {}

    for topo_short in args.topologies:
        graph_type = TOPOLOGY_MAP[topo_short]
        topo_label = f"{topo_short} N=100"

        print("\n" + "=" * 60)
        print(f"TOPOLOGY: {topo_label}  ({graph_type})")
        print(f"  num_companies={_probe_cfg.num_companies}, "
              f"num_steps={num_steps}, "
              f"total_timesteps={_probe_cfg.rl.total_timesteps:,}")
        print("=" * 60)

        cfg = build_config(graph_type)
        topo_dir = OUTPUT_DIR / graph_type

        results = run_topology(
            cfg=cfg,
            topo_dir=topo_dir,
            seed=args.seed,
            skip_training=args.skip_training,
        )
        all_results[topo_label] = results

        print_topology_table(topo_label, results)

        save_plots(topo_short, results, num_steps)

    if len(all_results) > 1:
        print_all_tables(all_results)

    print_statistical_tests(all_results)
    print_gnn_advantage_summary(all_results)

    elapsed = str(timedelta(seconds=int(time.time() - start)))
    print(f"\nAll done in {elapsed} ({datetime.now().strftime('%H:%M:%S')})")
    print(f"Report saved to: {report_path}")

    tee.close()


if __name__ == "__main__":
    main()
