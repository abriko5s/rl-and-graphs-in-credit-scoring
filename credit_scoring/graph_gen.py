"""
Graph Generator
Creates one of three available types of graphs:
  1. Barasabi-Albert (my attempt to model realistic supply-chain)
  https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model
  2. Erdos-Renyi (random edges)
  https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model
  3. Watts-Strogatz (clustering)
  https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from credit_scoring.config import Config
from credit_scoring.entities import CompanyNode, SupplyEdge


def generate_companies(config: Config, rng: np.random.Generator) -> list[CompanyNode]:
    """
    Генерируем компании
    50% healthy (equity 30-70% от активов)
    30% middle (equity 10-30% от активов)
    20% risky (equity 0-10% от активов или на грани дефолта)
    """
    num_risky = max(1, int(config.num_sectors * config.risky_sector_fraction))
    risky_sectors = set(range(config.num_sectors - num_risky, config.num_sectors))

    companies = []
    for i in range(config.num_companies):
        lo, hi = config.assets_range
        if getattr(config, "assets_distribution", "uniform") == "normal":
            mu = (lo + hi) / 2
            sigma = (hi - lo) / 4
            assets = float(np.clip(rng.normal(mu, sigma), lo * 0.3, hi * 2.0))
        else:
            assets = rng.uniform(lo, hi)
        sector = int(rng.integers(0, config.num_sectors))
        company_type = rng.random()
        if company_type < 0.5:
            liab_ratio = rng.uniform(0.3, 0.6)
        elif company_type < 0.8:
            liab_ratio = rng.uniform(0.6, 0.8)
        else:
            liab_ratio = rng.uniform(0.8, 0.95)

        burn_rate = rng.uniform(*config.burn_rate_range)
        productivity = rng.uniform(*config.productivity_range)

        # Конфиг для рисковых компаний
        if sector in risky_sectors:
            burn_rate *= config.risky_burn_rate_multiplier
            productivity = 1.0 + (productivity - 1.0) * config.risky_productivity_multiplier

        companies.append(CompanyNode(
            id=i,
            sector=sector,
            assets=assets,
            total_liabilities=assets * liab_ratio,
            burn_rate=burn_rate,
            productivity=productivity,
            liquidation_cost=config.liquidation_cost,
        ))
    return companies


def generate_graph(
        config: Config,
        rng: np.random.Generator,
        companies: list[CompanyNode] | None = None,
) -> nx.DiGraph:
    """
    Генерит обычный граф и превращает в ориентированный.
    Тип графа лежит в config.graph_type:
    1. barabasi_albert (m = ba_edges_per_node)
    2. erdos_renyi (p = er_prob)
    3. watts_strogatz (k, p)
    """
    n = config.num_companies
    seed = int(rng.integers(0, 2 ** 31))

    if config.graph_type == "barabasi_albert":
        G_undirected = nx.barabasi_albert_graph(n, config.ba_edges_per_node, seed=seed)
    elif config.graph_type == "erdos_renyi":
        G_undirected = nx.erdos_renyi_graph(n, config.er_prob, seed=seed, directed=False)
    elif config.graph_type == "watts_strogatz":
        G_undirected = nx.watts_strogatz_graph(n, config.ws_k, config.ws_p, seed=seed)
    else:
        raise ValueError(f"Unknown graph type: {config.graph_type}")

    if companies is not None:
        sector_groups: dict[int, list[int]] = {}
        for c in companies:
            sector_groups.setdefault(c.sector, []).append(c.id)

        for sector_nodes in sector_groups.values():
            if len(sector_nodes) < 2:
                continue
            # 15% шанс дополнительного ребра
            for i, u in enumerate(sector_nodes):
                for v in sector_nodes[i + 1:]:
                    if not G_undirected.has_edge(u, v) and rng.random() < 0.15:
                        G_undirected.add_edge(u, v)

    # Конвертируем в направленный граф: каждое ребро (u, v) становится
    # направленным ребром с debt_amount
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u, v in G_undirected.edges():
        debt = rng.uniform(*config.debt_amount_range)
        if rng.random() < 0.5:
            G.add_edge(u, v, debt_amount=debt)
        else:
            G.add_edge(v, u, debt_amount=debt)
    return G


def build_supply_edges(G: nx.DiGraph) -> list[SupplyEdge]:
    """
    Достать список обязательств из графа
    """
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append(SupplyEdge(
            source_id=u,
            target_id=v,
            debt_amount=data["debt_amount"],
        ))
    return edges


def generate_economy(
        config: Config, seed: int | None = None
) -> tuple[list[CompanyNode], nx.DiGraph, list[SupplyEdge]]:
    """
    Генерим среду со всеми участниками
    Public:
        config: конфигурация симулятора
        seed: seed
    Return:
        (companies, graph, edges)
    """
    rng = np.random.default_rng(seed)
    companies = generate_companies(config, rng)
    G = generate_graph(config, rng, companies=companies)
    edges = build_supply_edges(G)

    # обновление total_liabilities
    for company in companies:
        incoming_debt = sum(e.debt_amount for e in edges if e.target_id == company.id)
        company.total_liabilities = max(company.total_liabilities, incoming_debt) # чтобы с самого начала компания не стала банкротом
    return companies, G, edges
