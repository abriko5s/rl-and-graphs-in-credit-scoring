"""
Environment and Experiments configuration
Using .yaml files for configs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import yaml


@dataclass
class RLConfig:
    """
    Гиперпараметры для агентов
    """
    algorithm: str = "PPO"  # PPO | A2C | DQN (deprecated)
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4  # classic xD
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    n_epochs: int = 10  # PPO only
    seed: int = 42


@dataclass
class GNNConfig:
    """
    Гиперпараметры для графовой нейронки
    """
    hidden_dim: int = 64
    num_layers: int = 2
    conv_type: str = "GCN"  # GCN / GAT
    dropout: float = 0.1


@dataclass
class CatBoostConfig:
    """
    Гиперпараметры для Catboost
    """
    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    approval_threshold: float = 0.5


@dataclass
class Config:
    """
    Все самое нужное

    Attributes:
        num_companies: количество компаний в графе
        num_sectors: количество секторов
        num_steps: количество шагов (заявок) за один эпизод
        interest_rate: годовая процентная ставка по кредитам
        graph_type: тип графа
        assets_range: диапазон начальных активов компании
        burn_rate_range: скорость трат компании за шаг
        productivity_range: сколько компания зарабатывает
        liquidation_cost: доля активов, возвращаемая при банкротстве
        loan_amount_range: диапазон сумм заявок
        loan_term_range: диапазон срокоы кредитов (шаги!!!)
        contagion_fraction: доля долга, теряемая контрагентом при дефолте
        debt_amount_range: диапазон долговых обязательств между компаниями
        shock_probability: вероятность шока на каждом шаге
        shock_asset_loss: доля активов, теряемая при шоке
    """

    # env
    num_companies: int = 50
    num_sectors: int = 5
    num_steps: int = 200
    interest_rate: float = 0.05

    # graph
    graph_type: str = "barabasi_albert"
    ba_edges_per_node: int = 3
    er_prob: float = 0.1
    ws_k: int = 4
    ws_p: float = 0.3

    # companies
    assets_range: Tuple[float, float] = (50.0, 500.0)
    burn_rate_range: Tuple[float, float] = (0.01, 0.05)
    productivity_range: Tuple[float, float] = (0.8, 1.2)
    liquidation_cost: float = 0.4

    # applications
    loan_amount_range: Tuple[float, float] = (10.0, 100.0)
    loan_term_range: Tuple[int, int] = (5, 20)

    # contagion
    contagion_fraction: float = 0.3
    debt_amount_range: Tuple[float, float] = (5.0, 50.0)

    # rewards
    rejection_penalty: float = 0.3  # штраф за отказ (доля от суммы которую хотят занять)
    loan_liability_passthrough: float = 0.5  # доля кредита, увеличивающая liabilities
    contagion_leverage_amplifier: float = 1.0  # bank_debt усиливает заражение
    default_cost: float = 20.0  # фиксированная стоимость дефолта для банка

    # cooldown
    cooldown_max_steps: int = 30

    # shocks
    # 1 comp
    shock_probability: float = 0.0
    shock_asset_loss: float = 0.5
    # sector
    sector_shock_probability: float = 0.0
    sector_shock_asset_loss: float = 0.2
    # macro
    macro_shock_probability: float = 0.0
    macro_shock_asset_loss: float = 0.1

    # distr
    assets_distribution: str = "normal"  # "uniform" или "normal"

    # доля risky
    risky_sector_fraction: float = 0.2
    # используем повышенный burn_rate и productivity (например, стартапы)
    risky_burn_rate_multiplier: float = 2.5
    risky_productivity_multiplier: float = 1.5

    ablation_no_edges: bool = False  # GNN не видит ребра при true

    # все конфиги
    rl: RLConfig = field(default_factory=RLConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Грузим конфиг из yaml файла
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        rl_data = data.pop("rl", {})
        gnn_data = data.pop("gnn", {})
        cb_data = data.pop("catboost", {})
        for key in data:
            if key.endswith("_range") and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(
            **data,
            rl=RLConfig(**rl_data),
            gnn=GNNConfig(**gnn_data),
            catboost=CatBoostConfig(**cb_data),
        )
