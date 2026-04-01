"""
Graphics stuff
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    SB3 Callback
    """

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "cumulative_pnl" in info:
                self.logger.record("env/cumulative_pnl", info["cumulative_pnl"])
                self.logger.record("env/step_pnl", info["step_pnl"])
                self.logger.record("env/alive_companies", info["alive_companies"])
                self.logger.record("env/total_defaults", info["total_defaults"])
                self.logger.record("env/approval_rate", info["approval_rate"])
                self.logger.record("env/borrower_default_rate", info.get("borrower_default_rate", 0))

                self.logger.record("env/sharpe_ratio", info.get("sharpe_ratio", 0))
                self.logger.record("env/roa", info.get("roa", 0))
                self.logger.record("env/survival_rate", info.get("survival_rate", 0))
                self.logger.record("env/max_cascade_depth", info.get("max_cascade_depth", 0))
                self.logger.record("env/total_loans_issued", info.get("total_loans_issued", 0))

                self.logger.record("env/pnl_per_step", info.get("pnl_per_step", 0))
                self.logger.record("env/pnl_per_company", info.get("pnl_per_company", 0))
        return True


def plot_comparison(
    results: dict[str, list[float]],
    title: str = "Cumulative P&L Comparison",
    save_path: str | Path | None = None,
    num_active_steps: int | None = None,
) -> None:
    """
    Сравнение pnl

    Args:
        results: {agent_name: [pnl_per_step]}
        title
        save_path
        num_active_steps
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for name, pnl_history in results.items():
        total_len = len(pnl_history)
        cumulative = np.cumsum(pnl_history)

        if num_active_steps and total_len > num_active_steps:
            line, = ax.plot(
                range(num_active_steps), cumulative[:num_active_steps], label=name,
            )
            ax.plot(
                range(num_active_steps - 1, total_len),
                cumulative[num_active_steps - 1:],
                linestyle="--", color=line.get_color(), alpha=0.5,
            )
        elif num_active_steps and total_len < num_active_steps:
            ax.plot(range(total_len), cumulative, label=name)
            ax.plot(
                total_len - 1, cumulative[-1],
                "x", color="black", markersize=10, markeredgewidth=2,
            )
        else:
            ax.plot(range(total_len), cumulative, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_sector_analysis(
    sector_approvals: dict[str, dict[int, int]],
    sector_pnl: dict[str, dict[int, float]],
    save_path: str | Path | None = None,
) -> None:
    """
    Анализ для H2
    Args:
        sector_approvals: {agent_name: {sector: count}}
        sector_pnl: {agent_name: {sector: pnl}}
    """
    agents = list(sector_approvals.keys())
    sectors = sorted(set().union(*(d.keys() for d in sector_approvals.values())))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(sectors))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        approvals = [sector_approvals[agent].get(s, 0) for s in sectors]
        ax1.bar(x + i * width, approvals, width, label=agent)

    ax1.set_xlabel("Sector")
    ax1.set_ylabel("Approvals")
    ax1.set_title("Approvals by Sector")
    ax1.set_xticks(x + width * (len(agents) - 1) / 2)
    ax1.set_xticklabels([f"S{s}" for s in sectors])
    ax1.legend()

    for i, agent in enumerate(agents):
        pnl = [sector_pnl[agent].get(s, 0) for s in sectors]
        ax2.bar(x + i * width, pnl, width, label=agent)

    ax2.set_xlabel("Sector")
    ax2.set_ylabel("P&L")
    ax2.set_title("P&L by Sector")
    ax2.set_xticks(x + width * (len(agents) - 1) / 2)
    ax2.set_xticklabels([f"S{s}" for s in sectors])
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_degree_vs_approval(
    degrees: list[int],
    approval_rates: list[float],
    agent_name: str = "GNN Agent",
    save_path: str | Path | None = None,
) -> None:
    """
    График для H3
    Args:
        degrees
        approval_rates
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(degrees, approval_rates, alpha=0.5)
    z = np.polyfit(degrees, approval_rates, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(degrees), max(degrees), 100)
    ax.plot(x_line, p(x_line), "r--", label=f"Trend (slope={z[0]:.4f})")

    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Approval Rate")
    ax.set_title(f"{agent_name}: Approval Rate vs Node Degree")
    ax.legend()
    ax.grid(True, alpha=0.3)

    corr = np.corrcoef(degrees, approval_rates)[0, 1]
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_percentage_profit(
    results: dict[str, dict],
    title: str = "Percentage Profit (ROA %)",
    save_path: str | Path | None = None,
) -> None:
    """
    график ROA = cumulative_pnl / total_loans_issued * 100%

    Args:
        results: {agent_name: eval_results}
        title
        save_path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    agents = list(results.keys())
    roa_values = [results[a].get("mean_roa", 0) * 100 for a in agents]
    pnl_values = [results[a].get("mean_pnl", 0) for a in agents]
    pnl_std = [results[a].get("std_pnl", 0) for a in agents]

    # ROA %
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(agents)))
    sorted_idx = np.argsort(roa_values)[::-1]
    sorted_agents = [agents[i] for i in sorted_idx]
    sorted_roa = [roa_values[i] for i in sorted_idx]
    bar_colors = [("green" if v > 0 else "red") for v in sorted_roa]

    ax1.barh(range(len(sorted_agents)), sorted_roa, color=bar_colors, alpha=0.7)
    ax1.set_yticks(range(len(sorted_agents)))
    ax1.set_yticklabels(sorted_agents)
    ax1.set_xlabel("ROA (%)")
    ax1.set_title("Return on Assets (%)")
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(True, alpha=0.3, axis="x")

    # PnL
    sorted_pnl = [pnl_values[i] for i in sorted_idx]
    sorted_std = [pnl_std[i] for i in sorted_idx]
    ax2.barh(range(len(sorted_agents)), sorted_pnl, xerr=sorted_std,
             color=bar_colors, alpha=0.7, capsize=3)
    ax2.set_yticks(range(len(sorted_agents)))
    ax2.set_yticklabels(sorted_agents)
    ax2.set_xlabel("Cumulative P&L")
    ax2.set_title("Mean Cumulative P&L")
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


