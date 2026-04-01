"""
how to use:
  python run.py train --agent gnn
  python run.py train --agent mlp
  python run.py train --agent catboost
  python run.py eval --agent gnn
  python run.py experiment h1
  python run.py experiment h1-scale #using BA
  python run.py experiment h1-full
  python run.py experiment h2
  python run.py experiment h3
  python run.py experiment all
  python run.py stress --agent gnn
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from credit_scoring.config import Config


class TeeWriter:
    """
    logging stuff
    """

    def __init__(self, filepath: str | Path):
        self.terminal = sys.stdout
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
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
        self.log.flush()

    def close(self) -> None:
        self.log.close()
        sys.stdout = self.terminal


def main():
    parser = argparse.ArgumentParser(
        description="RL + GNN Credit Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default="configs/heavy.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory for models and results",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # train
    train_parser = subparsers.add_parser("train", help="Train an agent")
    train_parser.add_argument(
        "--agent", type=str, required=True,
        choices=["gnn", "mlp", "catboost"],
        help="Agent type to train",
    )

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "--agent", type=str, required=True,
        choices=["gnn", "mlp", "catboost", "heuristic"],
    )
    eval_parser.add_argument("--episodes", type=int, default=10)

    # experiment
    exp_parser = subparsers.add_parser("experiment", help="Run an experiment")
    exp_parser.add_argument(
        "name", type=str,
        choices=["h1", "h1-scale", "h1-full", "h2", "h3", "all"],
        help=(
            "h1 = topologies (BA/ER/WS), "
            "h1-scale = graph sizes, "
            "h1-full = topologies + scale, "
            "all = H1 + H2 + H3"
        ),
    )
    exp_parser.add_argument("--episodes", type=int, default=10)
    exp_parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip training, load existing models",
    )
    exp_parser.add_argument(
        "--graph-sizes", type=int, nargs="+", default=[5, 10, 20, 50],
        help="Graph sizes for h1-scale (default: 5 10 20 50)",
    )

    # stress
    stress_parser = subparsers.add_parser("stress", help="Stress-test an agent")
    stress_parser.add_argument(
        "--agent", type=str, required=True,
        choices=["gnn", "mlp", "catboost", "heuristic"],
    )
    stress_parser.add_argument("--episodes", type=int, default=10)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = Config.from_yaml(args.config)

    if args.command == "train":
        train(args, config)
    elif args.command == "eval":
        evaluate(args, config)
    elif args.command == "experiment":
        experiment(args, config)
    elif args.command == "stress":
        stress_test(args, config)


# Commands
def train(args, config: Config):
    if args.agent in ("gnn", "mlp"):
        from credit_scoring.training.train_rl import train_agent
        train_agent(args.agent, config, output_dir=args.output, seed=args.seed)
    elif args.agent == "catboost":
        from credit_scoring.training.train_classical import train_classical_baseline
        train_classical_baseline(config, output_dir=args.output, seed=args.seed)


def evaluate(args, config: Config):
    from credit_scoring.evaluation.evaluate import evaluate_heuristic_agent, evaluate_rl_agent

    if args.agent == "gnn":
        from credit_scoring.agents.gnn_agent import load_gnn_agent
        conv = config.gnn.conv_type.lower()
        algo = config.rl.algorithm.lower()
        suffix = f"gnn_{conv}_{algo}"
        model = load_gnn_agent(
            Path(args.output) / suffix / f"{suffix}_final", config, seed=args.seed,
            algorithm=config.rl.algorithm,
        )
        results = evaluate_rl_agent(
            model, config, num_episodes=args.episodes, seed=args.seed,
        )
    elif args.agent == "mlp":
        from credit_scoring.agents.mlp_agent import load_mlp_agent
        algo = config.rl.algorithm.lower()
        suffix = f"mlp_{algo}"
        model = load_mlp_agent(
            Path(args.output) / suffix / f"{suffix}_final", config, seed=args.seed,
            algorithm=config.rl.algorithm,
        )
        results = evaluate_rl_agent(
            model, config, num_episodes=args.episodes, seed=args.seed,
            use_flat_env=True,
        )
    elif args.agent == "catboost":
        from credit_scoring.baselines.classical_ml import CatBoostAgent
        agent = CatBoostAgent.load(
            Path(args.output) / "catboost" / "catboost_model.cbm",
        )
        results = evaluate_heuristic_agent(
            agent, config, num_episodes=args.episodes, seed=args.seed,
        )
    elif args.agent == "heuristic":
        from credit_scoring.agents.heuristic_agent import EquityThresholdAgent
        agent = EquityThresholdAgent(threshold=50.0)
        results = evaluate_heuristic_agent(
            agent, config, num_episodes=args.episodes, seed=args.seed,
        )
    # vibe coded for beautiful output
    print(f"\nEvaluation Results ({args.agent}):")
    print(f"  Mean P&L:        {results['mean_pnl']:.2f} +/- {results['std_pnl']:.2f}")
    print(f"  Mean Defaults:   {results['mean_defaults']:.1f}")
    print(f"  Approval Rate:   {results['mean_approval_rate']:.1%}")


def experiment(args, config: Config):
    report_path = Path(args.output) / "report.txt"
    tee = TeeWriter(report_path)
    sys.stdout = tee

    start = time.time()
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    name = args.name
    if name == "h1-full" or name == "h1":
        names = ["h1-scale"]
    elif name == "all":
        names = ["h1-scale", "h2", "h3"]
    else:
        names = [name]

    for n in names:
        if n == "h1":
            run_h1_topologies(args, config)
        elif n == "h1-scale":
            run_h1_scale(args, config)
        elif n == "h2":
            run_h2(args, config)
        elif n == "h3":
            run_h3(args, config)

    elapsed = str(timedelta(seconds=int(time.time() - start)))
    print(f"\nAll done in {elapsed} ({datetime.now().strftime('%H:%M:%S')})")
    print(f"Report saved to: {report_path}")

    tee.close()


def stress_test(args, config: Config):
    from credit_scoring.evaluation.stress_test import stress_test_with_shocks

    print(f"\nStress-testing {args.agent} agent...")

    if args.agent == "gnn":
        from credit_scoring.agents.gnn_agent import load_gnn_agent
        conv = config.gnn.conv_type.lower()
        algo = config.rl.algorithm.lower()
        suffix = f"gnn_{conv}_{algo}"
        model = load_gnn_agent(
            Path(args.output) / suffix / f"{suffix}_final", config, seed=args.seed,
            algorithm=config.rl.algorithm,
        )
        stress_test_with_shocks(
            None, config, agent_type="gnn", model=model,
            num_episodes=args.episodes, seed=args.seed,
        )
    elif args.agent == "mlp":
        from credit_scoring.agents.mlp_agent import load_mlp_agent
        algo = config.rl.algorithm.lower()
        suffix = f"mlp_{algo}"
        model = load_mlp_agent(
            Path(args.output) / suffix / f"{suffix}_final", config, seed=args.seed,
            algorithm=config.rl.algorithm,
        )
        stress_test_with_shocks(
            None, config, agent_type="mlp", model=model,
            num_episodes=args.episodes, seed=args.seed,
        )
    elif args.agent == "catboost":
        from credit_scoring.baselines.classical_ml import CatBoostAgent
        agent = CatBoostAgent.load(
            Path(args.output) / "catboost" / "catboost_model.cbm",
        )
        stress_test_with_shocks(
            agent, config, agent_type="catboost",
            num_episodes=args.episodes, seed=args.seed,
        )
    elif args.agent == "heuristic":
        from credit_scoring.agents.heuristic_agent import EquityThresholdAgent
        agent = EquityThresholdAgent(threshold=50.0)
        stress_test_with_shocks(
            agent, config, agent_type="heuristic",
            num_episodes=args.episodes, seed=args.seed,
        )


# Runners

def run_h1_topologies(args, config: Config):
    """
    Сравниваем графы
    """
    from credit_scoring.experiments.h1_graph_vs_tabular import run_h1_experiment

    print("=" * 60)
    print("H1: TOPOLOGY SWEEP (BA, ER, WS)")
    print("=" * 60)

    run_h1_experiment(
        config,
        output_dir=f"{args.output}/h1",
        num_eval_episodes=args.episodes,
        seed=args.seed,
        skip_training=args.skip_training,
    )


def run_h1_scale(args, config: Config):
    """
    Сравниваем размеры
    """
    from credit_scoring.experiments.h1_graph_vs_tabular import run_h1_scale_experiment

    print("=" * 60)
    print(f"H1: SCALE SWEEP ({args.graph_sizes})")
    print("=" * 60)

    run_h1_scale_experiment(
        config,
        output_dir=f"{args.output}/h1_scale",
        graph_sizes=args.graph_sizes,
        num_eval_episodes=args.episodes,
        seed=args.seed,
        skip_training=args.skip_training,
    )


def run_h2(args, config: Config):
    """
    Смотрим секторы
    """
    from credit_scoring.experiments.h2_exploration import run_h2_experiment

    print("=" * 60)
    print("H2: SECTOR EXPLORATION")
    print("=" * 60)

    run_h2_experiment(
        config,
        output_dir=f"{args.output}/h2",
        num_eval_episodes=args.episodes,
        seed=args.seed,
        skip_training=args.skip_training,
    )


def run_h3(args, config: Config):
    """
    Смотрим дискриминациию
    """
    from credit_scoring.experiments.h3_discrimination import run_h3_experiment

    print("=" * 60)
    print("H3: STRUCTURAL DISCRIMINATION")
    print("=" * 60)

    run_h3_experiment(
        config,
        output_dir=f"{args.output}/h3",
        num_eval_episodes=args.episodes,
        seed=args.seed,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()
