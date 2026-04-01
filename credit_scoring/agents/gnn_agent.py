"""
GNN-agent
  1. GNN encoder (GCN or GAT) #fortest
  2. Concatenate embeddings into vector
  3. MLP-head returns approve/reject and value
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv, GCNConv

from credit_scoring.config import Config, GNNConfig
from credit_scoring.env import APPLICATION_FEATURE_DIM, NODE_FEATURE_DIM, CreditScoringEnv


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extraction using GNN
    """

    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            gnn_config: GNNConfig,
    ):
        # features_dim: applicant_emb + graph_context + app_emb
        hidden = gnn_config.hidden_dim
        features_dim = hidden + hidden + hidden
        super().__init__(observation_space, features_dim=features_dim)

        self.gnn_config = gnn_config

        # gnn layers
        ConvClass = GATConv if gnn_config.conv_type == "GAT" else GCNConv

        self.convs = nn.ModuleList()
        self.convs.append(ConvClass(NODE_FEATURE_DIM, hidden))
        for _ in range(gnn_config.num_layers - 1):
            self.convs.append(ConvClass(hidden, hidden))

        self.norms = nn.ModuleList()
        for _ in range(gnn_config.num_layers):
            self.norms.append(nn.LayerNorm(hidden))

        self.dropout = nn.Dropout(gnn_config.dropout)
        self.activation = nn.ReLU()

        # linear
        self.app_encoder = nn.Sequential(
            nn.Linear(APPLICATION_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        Args:
            observations: Dict w/ keys: node_features, edge_index, edge_attr, num_edges, application_features
        Returns:
            [batch, 3 * hidden_dim]:
              1 embedding from GNN
              2 global graph condition
              3 encoded application features
        """
        node_features = observations["node_features"]  # [batch, N, F]
        edge_index_padded = observations["edge_index"]  # [batch, 2, max_E]
        num_edges = observations["num_edges"]  # [batch, 1]
        applicant_ids = observations["applicant_id"]  # [batch, 1]
        app_features = observations["application_features"]  # [batch, app_dim]

        batch_size = node_features.shape[0]

        applicant_embeddings = []
        graph_contexts = []

        for b in range(batch_size):
            # edges
            ne = int(num_edges[b, 0].item())
            x = node_features[b]  # [N, F]
            ei = edge_index_padded[b, :, :ne].long()  # [2, ne]

            # Прогоняем через GNN с residual connections и layer norm
            for i, conv in enumerate(self.convs):
                x_new = conv(x, ei)
                x_new = self.norms[i](x_new)
                x_new = self.activation(x_new)
                x_new = self.dropout(x_new)
                # Residual connection (если размерности совпадают)
                if x.shape == x_new.shape:
                    x = x + x_new
                else:
                    x = x_new

            # embed from gnn
            # applicant_id точно указывает на нужный узел в графе
            aid = int(applicant_ids[b, 0].item())
            aid = min(aid, x.shape[0] - 1)
            applicant_emb = x[aid]  # [hidden]
            applicant_embeddings.append(applicant_emb)

            # global info
            alive_mask = node_features[b, :, 6] < 0.5
            if alive_mask.any():
                graph_ctx = x[alive_mask].mean(dim=0)
            else:
                graph_ctx = x.mean(dim=0)
            graph_contexts.append(graph_ctx)

        applicant_emb = torch.stack(applicant_embeddings)  # [batch, hidden]
        graph_ctx = torch.stack(graph_contexts)  # [batch, hidden]

        # application features
        app_emb = self.app_encoder(app_features)  # [batch, hidden]

        return torch.cat([applicant_emb, graph_ctx, app_emb], dim=-1)


def _get_policy_kwargs(config: Config) -> dict:
    return {
        "features_extractor_class": GNNFeatureExtractor,
        "features_extractor_kwargs": {"gnn_config": config.gnn},
        "net_arch": dict(pi=[128, 128], vf=[128, 128]),
    }


def create_gnn_agent(
        config: Config, seed: int | None = None, algorithm: str | None = None,
) -> PPO | A2C:
    """
    Создает агента
    Args:
        config: конфиг
        seed: seed
        algorithm: PPO / A2C (None = из конфига)
    Returns:
        model: init агента
    """
    env = CreditScoringEnv(config=config, seed=seed)
    rl = config.rl
    actual_seed = rl.seed if seed is None else seed
    algo = algorithm or rl.algorithm

    policy_kwargs = _get_policy_kwargs(config)

    if algo == "A2C":
        model = A2C(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=rl.learning_rate,
            n_steps=min(rl.n_steps, 64),
            gamma=rl.gamma,
            gae_lambda=rl.gae_lambda,
            ent_coef=rl.ent_coef,
            seed=actual_seed,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=rl.learning_rate,
            n_steps=rl.n_steps,
            batch_size=rl.batch_size,
            gamma=rl.gamma,
            gae_lambda=rl.gae_lambda,
            ent_coef=rl.ent_coef,
            n_epochs=rl.n_epochs,
            seed=actual_seed,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )
    return model


def load_gnn_agent(
        path: str | Path, config: Config, seed: int | None = None,
        algorithm: str | None = None,
) -> PPO | A2C:
    """
    Загрузить агента
    """
    env = CreditScoringEnv(config=config, seed=seed)
    custom_objects = {"policy_kwargs": _get_policy_kwargs(config)}
    algo = algorithm or config.rl.algorithm

    cls = A2C if algo == "A2C" else PPO # todo: аккуратно если буду добавлять новый алгос
    return cls.load(str(path), env=env, custom_objects=custom_objects)
