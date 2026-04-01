"""
Запуск H2 и H3 на графе N=100 (BA).
Использует уже обученные модели из outputs/n100/barabasi_albert/.

Использование:
  python run_n100_h2h3.py
  python run_n100_h2h3.py --episodes 20
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from credit_scoring.config import Config


def main():
    parser = argparse.ArgumentParser(description="H2 + H3 на N=100")
    parser.add_argument("--config", type=str, default="configs/n100.yaml")
    parser.add_argument("--output", type=str, default="outputs/n100_h2h3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    model_dir = Path("outputs/n100/barabasi_albert")

    # Проверяем что модели есть
    gnn_path = model_dir / "gnn_gcn_ppo" / "gnn_gcn_ppo_final.zip"
    mlp_path = model_dir / "mlp_ppo" / "mlp_ppo_final.zip"
    if not gnn_path.exists() or not mlp_path.exists():
        print(f"ERROR: модели не найдены в {model_dir}")
        print(f"  Ожидается: {gnn_path}")
        print(f"  Ожидается: {mlp_path}")
        print("Сначала запусти: python run_n100.py")
        sys.exit(1)

    start = time.time()
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Config: {args.config}  (N={config.num_companies})")
    print(f"Models: {model_dir}")
    print(f"Eval episodes: {args.episodes}")
    print()

    # ── H2 ──────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("H2: SECTOR EXPLORATION  (N=100, BA)")
    print("=" * 60)

    from credit_scoring.experiments.h2_exploration import run_h2_experiment
    run_h2_experiment(
        config,
        output_dir=str(out / "h2"),
        num_eval_episodes=args.episodes,
        seed=args.seed,
        skip_training=True,
        model_dir=str(model_dir),
    )

    # ── H3 ──────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("H3: STRUCTURAL DISCRIMINATION  (N=100, BA)")
    print("=" * 60)

    from credit_scoring.experiments.h3_discrimination import run_h3_experiment
    run_h3_experiment(
        config,
        output_dir=str(model_dir),   # загружает модели из этой же папки
        num_eval_episodes=args.episodes,
        seed=args.seed,
        skip_training=True,
    )
    # копируем картинки в выходную папку
    import shutil
    for png in model_dir.glob("h3_*.png"):
        shutil.copy(png, out / "h3" / png.name)
        print(f"Copied: {out / 'h3' / png.name}")

    elapsed = str(timedelta(seconds=int(time.time() - start)))
    print(f"\nAll done in {elapsed} ({datetime.now().strftime('%H:%M:%S')})")


if __name__ == "__main__":
    main()
