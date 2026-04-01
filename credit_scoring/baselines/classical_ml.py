"""
classic CatBoost
    1. Generate data from the environment
    2. Collect pairs (company account, defaults in K steps)
    3. Learn CatBoost classifier: P(default | features)
    4. approve if P(default) < threshold
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from credit_scoring.config import Config
from credit_scoring.env import FlatCreditScoringEnv

# hyperparam!
DEFAULT_HORIZON = 20


def collect_dataset(
        config: Config,
        num_episodes: int = 50,
        seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Собираем датасет из среды

    Args:
        config: конфигурация
        num_episodes: количество эпизодов для сбора
        seed: seed
    Returns:
        X: [n_samples, n_features] -- фичи
        y: [n_samples] -- 1 если компания дефолтнула в пределах горизонта
    """
    all_features = []
    all_labels = []

    # Стратегии сбора данных (больше selective — экономика живёт дольше,
    # больше наблюдений и меньше шума от массовых дефолтов):
    # selective: хотим equity_ratio > 0.3
    # moderate: одобряем половину случайно
    # aggressive: approve all
    strategies = [
        ("selective", 0.3),
        ("selective", 0.3),
        ("moderate", 0.5),
        ("aggressive", 1.0),
    ]

    for ep in range(num_episodes):
        ep_seed = seed + ep
        strategy_name, approve_prob = strategies[ep % len(strategies)]
        rng = np.random.default_rng(ep_seed + 10000)

        env = FlatCreditScoringEnv(config=config, seed=ep_seed)
        obs, _ = env.reset(seed=ep_seed)
        done = False

        episode_data = []
        default_tracker: dict[int, int] = {}  # {company_id: step_of_default}
        step = 0

        while not done:
            app_id = env.sim.current_application.applicant_id
            episode_data.append({
                "obs": obs.copy(),
                "applicant_id": app_id,
                "step": step,
            })

            if strategy_name == "selective":
                equity = obs[3]  # equity в flat obs
                assets = obs[0]
                action = 1 if equity / max(1, assets) > approve_prob else 0
            else:
                action = 1 if rng.random() < approve_prob else 0

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
            # defaults tracking
            for cid, company in env.sim.companies.items():
                if company.is_defaulted and cid not in default_tracker:
                    default_tracker[cid] = step

        # labeling
        for record in episode_data:
            aid = record["applicant_id"]
            t = record["step"]
            ds = default_tracker.get(aid, float("inf"))
            label = 1.0 if ds <= t + DEFAULT_HORIZON else 0.0

            all_features.append(record["obs"])
            all_labels.append(label)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)

    return X, y


def train_catboost(
        X: np.ndarray,
        y: np.ndarray,
        config: Config,
) -> CatBoostClassifier:
    """
    Обучаем
    """
    cb = config.catboost

    X_df = pd.DataFrame(X, columns=[
        "assets", "total_liabilities", "bank_debt", "equity",
        "burn_rate", "productivity", "is_defaulted", "sector",
        "loan_amount", "loan_term", "expected_return",
    ])
    X_df["sector"] = X_df["sector"].astype(int).astype(str)

    model = CatBoostClassifier(
        iterations=cb.iterations,
        learning_rate=cb.learning_rate,
        depth=cb.depth,
        cat_features=["sector"],
        verbose=50,
        random_seed=config.rl.seed,
        auto_class_weights="Balanced",
    )

    model.fit(X_df, y)
    return model


class CatBoostAgent:
    """
    Дефолт approve если P(default) < threshold
    """

    def __init__(self, model: CatBoostClassifier, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, obs: np.ndarray, sim=None) -> int:
        """
        Принять решение по заявке
        """
        X_df = pd.DataFrame([obs], columns=[
            "assets", "total_liabilities", "bank_debt", "equity",
            "burn_rate", "productivity", "is_defaulted", "sector",
            "loan_amount", "loan_term", "expected_return",
        ])
        X_df["sector"] = X_df["sector"].astype(int).astype(str)

        prob_default = self.model.predict_proba(X_df)[0, 1]
        return 1 if prob_default < self.threshold else 0

    def save(self, path: str | Path) -> None:
        """
        Сохранить модель
        """
        self.model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.5) -> "CatBoostAgent":
        """
        Загрузить модель
        """
        model = CatBoostClassifier()
        model.load_model(str(path))
        return cls(model=model, threshold=threshold)


class CatBoostAdaptiveAgent:
    """
    CatBoost с адаптивным threshold

    Такая модель предсказывает P(default | features). Во время кризиса
    P(default) повышается для всех компаний.
    Адаптация описывается как: threshold = base / survival_rate

    attr:
        model: обученный CatBoostClassifier
        base_threshold: начальный порог для здоровой экономики
        max_threshold: верхний предел threshold при кризисе
        window: окно для скользящего среднего survival rate
    """

    def __init__(
            self,
            model: CatBoostClassifier,
            base_threshold: float = 0.3,
            max_threshold: float = 0.7,
            window: int = 20,
    ):
        self.model = model
        self.base_threshold = base_threshold
        self.max_threshold = max_threshold
        self.window = window
        self._survival_history: list[float] = []

    def predict(self, obs: np.ndarray, sim=None) -> int:
        """
        Принять решение с адаптивным threshold
        """
        # track survival rate
        if sim is not None:
            sr = sim.get_survival_rate()
            self._survival_history.append(sr)

        if len(self._survival_history) >= self.window:
            recent_sr = np.mean(self._survival_history[-self.window:])
        elif self._survival_history:
            recent_sr = np.mean(self._survival_history)
        else:
            recent_sr = 1.0

        clamped_sr = max(recent_sr, 0.3)  # threshold for divider
        adaptive_threshold = min(
            self.base_threshold / clamped_sr,
            self.max_threshold,
        )

        # predict
        X_df = pd.DataFrame([obs], columns=[
            "assets", "total_liabilities", "bank_debt", "equity",
            "burn_rate", "productivity", "is_defaulted", "sector",
            "loan_amount", "loan_term", "expected_return",
        ])
        X_df["sector"] = X_df["sector"].astype(int).astype(str)

        prob_default = self.model.predict_proba(X_df)[0, 1]
        return 1 if prob_default < adaptive_threshold else 0

    def reset(self) -> None:
        """
        Сбросить для некст прогона
        """
        self._survival_history = []
