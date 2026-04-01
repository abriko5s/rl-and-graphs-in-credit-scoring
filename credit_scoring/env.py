"""
Gymnasium wrapper for MarketSimulator
Provides the standard Gymnasium interface Stable-Baselines3 correct work

Observation space:
  - node_features: matrix [num_companies, 7] - integers of all companies
  - edge_index: [2, num_edges] - Graph border indexes
  - edge_attr: [num_edges, 1] - debt_amount of each edge
  -application_features: [4] - details of the current credit application

Action space:
  0 = reject, 1 = approve
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from credit_scoring.config import Config
from credit_scoring.entities import CompanyNode
from credit_scoring.graph_gen import generate_economy
from credit_scoring.market import MarketSimulator

# фичи компании: assets, total_liabilities, bank_debt,
# equity, burn_rate, productivity, is_defaulted
NODE_FEATURE_DIM = 7

# фичи заявки: applicant_sector, amount, term, expected_return,
# applicant_equity, applicant_assets, applicant_bank_debt
APPLICATION_FEATURE_DIM = 7


class CreditScoringEnv(gym.Env):
    """
    Среда для обучения RL-агента кредитному скорингу

    На каждом шаге агент получает:
      - состояние графа экономики
      - текущую кредитную заявку
    и возвращает approve | reject
    Reward = step_pnl (процентный доход − потери от дефолтов)
    Эпизод заканчивается после num_steps шагов
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config | None = None, seed: int | None = None):
        super().__init__()

        self.config = config or Config()
        self._seed = seed
        self.sim: MarketSimulator | None = None

        # approve | reject
        self.action_space = spaces.Discrete(2)

        # get_graph_obs() для gnn
        n = self.config.num_companies
        max_edges = n * (n - 1)

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(
                low=-1e6, high=1e6,
                shape=(n, NODE_FEATURE_DIM),
                dtype=np.float32,
            ),
            "edge_index": spaces.Box(
                low=0, high=n - 1,
                shape=(2, max_edges),
                dtype=np.int64,
            ),
            "edge_attr": spaces.Box(
                low=0, high=1e6,
                shape=(max_edges, 1),
                dtype=np.float32,
            ),
            "num_edges": spaces.Box(
                low=0, high=max_edges,
                shape=(1,),
                dtype=np.int64,
            ),
            "applicant_id": spaces.Box(
                low=0, high=n - 1,
                shape=(1,),
                dtype=np.int64,
            ),
            "application_features": spaces.Box(
                low=-1e6, high=1e6,
                shape=(APPLICATION_FEATURE_DIM,),
                dtype=np.float32,
            ),
        })

    def _get_node_features(self) -> np.ndarray:
        """
        Матрица фичей компаний [num_companies, NODE_FEATURE_DIM]
        """
        features = np.zeros(
            (self.config.num_companies, NODE_FEATURE_DIM), dtype=np.float32
        )
        for company in self.sim.companies.values():
            features[company.id] = [
                company.assets,
                company.total_liabilities,
                company.bank_debt,
                company.equity(),
                company.burn_rate,
                company.productivity,
                float(company.is_defaulted),
            ]
        return features

    def _get_edge_data(self) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Индексы и атрибуты ребер
        Returns:
            edge_index: [2, num_edges] - source, target ids
            edge_attr: [num_edges, 1] - debt_amount
            num_edges: количество ребер
        """
        max_edges = self.config.num_companies * (self.config.num_companies - 1)
        edge_index = np.zeros((2, max_edges), dtype=np.int64)
        edge_attr = np.zeros((max_edges, 1), dtype=np.float32)

        # хардкод случая когда агент не видит ребра
        if self.config.ablation_no_edges:
            return edge_index, edge_attr, 0

        edges = list(self.sim.graph.edges(data=True))
        num_edges = len(edges)

        for i, (u, v, data) in enumerate(edges):
            edge_index[0, i] = u
            edge_index[1, i] = v
            edge_attr[i, 0] = data.get("debt_amount", 0.0)

        return edge_index, edge_attr, num_edges

    def _get_application_features(self) -> np.ndarray:
        """
        Вектор фичей текущей заявки
        """
        app = self.sim.current_application
        applicant = self.sim.companies[app.applicant_id]

        return np.array([
            float(applicant.sector),
            app.amount,
            float(app.term),
            app.expected_return,
            applicant.equity(),
            applicant.assets,
            applicant.bank_debt,
        ], dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        Собрать полное наблюдение для агента
        """
        edge_index, edge_attr, num_edges = self._get_edge_data()

        return {
            "node_features": self._get_node_features(),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_edges": np.array([num_edges], dtype=np.int64),
            "applicant_id": np.array(
                [self.sim.current_application.applicant_id], dtype=np.int64
            ),
            "application_features": self._get_application_features(),
        }

    def get_flat_obs(self) -> np.ndarray:
        """
        Плоское наблюдение для MLP-агента. Используются только фичи заявителя и заявки
        """
        app = self.sim.current_application
        applicant = self.sim.companies[app.applicant_id]

        return np.array([
            applicant.assets,
            applicant.total_liabilities,
            applicant.bank_debt,
            applicant.equity(),
            applicant.burn_rate,
            applicant.productivity,
            float(applicant.is_defaulted),
            float(applicant.sector),
            app.amount,
            float(app.term),
            app.expected_return,
        ], dtype=np.float32)

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Для перезапуска
        """
        super().reset(seed=seed)

        actual_seed = seed if seed is not None else self._seed
        companies, graph, edges = generate_economy(self.config, seed=actual_seed)
        rng = np.random.default_rng(actual_seed)

        self.sim = MarketSimulator(
            companies=companies,
            graph=graph,
            edges=edges,
            config=self.config,
            rng=rng,
        )

        self.sim.generate_application()
        obs = self._get_obs()
        info = self.sim.get_metrics()

        return obs, info

    def step(
            self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Шаг для среды
        action: 0 = reject, 1 = approve
        Returns:
            observation, reward, terminated, truncated, info
        """
        step_pnl = self.sim.process_step(action)

        terminated = self.sim.get_alive_count() == 0
        truncated = self.sim.current_step >= self.config.num_steps

        if terminated or truncated:
            self.sim.run_cooldown()
        else:
            self.sim.generate_application()

        obs = self._get_obs()
        info = self.sim.get_metrics()
        return obs, step_pnl, terminated, truncated, info


class FlatCreditScoringEnv(gym.Env):
    """
    Плоская версия среды для MLP и Catboost

    Observation space: Box — вектор фичей заявителя + заявки
    Используется для сравнения: показать, что GNN-агент лучше именно
    благодаря графовой информации.
    """

    metadata = {"render_modes": ["human"]}
    FLAT_OBS_DIM = 11

    def __init__(self, config: Config | None = None, seed: int | None = None):
        super().__init__()
        self.config = config or Config()
        self._seed = seed

        self._full_env = CreditScoringEnv(config=self.config, seed=seed)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self.FLAT_OBS_DIM,),
            dtype=np.float32,
        )

    def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self._full_env.reset(seed=seed, options=options)
        return self._full_env.get_flat_obs(), info

    def step(
            self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, reward, terminated, truncated, info = self._full_env.step(action)
        return self._full_env.get_flat_obs(), reward, terminated, truncated, info

    @property
    def sim(self) -> MarketSimulator:
        return self._full_env.sim
