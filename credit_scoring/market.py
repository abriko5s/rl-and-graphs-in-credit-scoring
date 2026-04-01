"""
Market Simulator

  1. Companies spend money (burn_rate) and generate revenue (productivity)
  2. Agent approves/rejects credit requests
  3. Approved loans increase the borrower’s assets and debt to the bank
  4. Companies with negative equity will default
  5. Default spreads by degree and contagion process starting
  6. Bank receives partial recovery in case of default
 7. Reward = interest income from credit losses - defaults loses
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import networkx as nx
import numpy as np

from credit_scoring.config import Config
from credit_scoring.entities import CompanyNode, LoanApplication, SupplyEdge


class MarketSimulator:
    """
    Симулятор экономики с графом компаний и финансовым заражением
    1. apply_burn_rate & productivity
    2. approve/reject
    3. process_loans
    4. apply_shock
    5. run_contagion_loop
    6. calculate_recovery

    graph: nx.DiGraph
    companies: {id: CompanyNode}
    current_application: текущая заявка
    step_pnl: PnL шага
    cumulative_pnl: накопленный PnL за эпизод
    current_step: номер шага
    """

    def __init__(
            self,
            companies: list[CompanyNode],
            graph: nx.DiGraph,
            edges: list[SupplyEdge],
            config: Config,
            rng: np.random.Generator | None = None,
    ):
        self.config = config
        self.rng = rng or np.random.default_rng()

        self._initial_companies = deepcopy(companies)
        self._initial_graph = graph.copy()
        self._initial_edges = deepcopy(edges)

        self.companies: dict[int, CompanyNode] = {c.id: c for c in deepcopy(companies)}
        self.graph: nx.DiGraph = graph.copy()
        self.edges: list[SupplyEdge] = deepcopy(edges)

        self.current_application: LoanApplication | None = None
        self.step_pnl: float = 0.0
        self.cumulative_pnl: float = 0.0
        self.current_step: int = 0
        self.total_defaults: int = 0
        self.total_approvals: int = 0
        self.total_rejections: int = 0

        self.borrowers: set[int] = set()

        # metrics
        self.sector_approvals: dict[int, int] = {s: 0 for s in range(config.num_sectors)}
        self.sector_requests: dict[int, int] = {s: 0 for s in range(config.num_sectors)}
        self.sector_pnl: dict[int, float] = {s: 0.0 for s in range(config.num_sectors)}
        self.total_loans_issued: float = 0.0
        self.max_cascade_depth: int = 0
        self.step_pnl_history: list[float] = []  # for plots!!!

    def reset(self) -> None:
        """
        Сброс между прогонами
        """
        self.companies = {c.id: deepcopy(c) for c in self._initial_companies}
        self.graph = self._initial_graph.copy()
        self.edges = deepcopy(self._initial_edges)
        self.current_application = None
        self.step_pnl = 0.0
        self.cumulative_pnl = 0.0
        self.current_step = 0
        self.total_defaults = 0
        self.total_approvals = 0
        self.total_rejections = 0
        self.sector_approvals = {s: 0 for s in range(self.config.num_sectors)}
        self.sector_requests = {s: 0 for s in range(self.config.num_sectors)}
        self.sector_pnl = {s: 0.0 for s in range(self.config.num_sectors)}
        self.borrowers = set()
        self.total_loans_issued = 0.0
        self.max_cascade_depth = 0
        self.step_pnl_history = []

    def generate_application(self) -> LoanApplication:
        """
        Генерируем заявку: берем случайную живую компанию, сумму и срок
        """
        alive = [c for c in self.companies.values() if not c.is_defaulted]
        if not alive:
            # при всех дефолтах заявка от нулевой компании (в дальнейшем не считаем)
            applicant = self.companies[0]
        else:
            applicant = self.rng.choice(alive)

        amount = self.rng.uniform(*self.config.loan_amount_range)
        lo, hi = self.config.loan_term_range
        term = int(self.rng.integers(lo, hi + 1))
        expected_return = amount * self.config.interest_rate * term

        self.current_application = LoanApplication(
            applicant_id=applicant.id,
            amount=amount,
            term=term,
            expected_return=expected_return,
        )
        # track requests per sector for normalized approval rate
        if alive:
            self.sector_requests[applicant.sector] += 1
        return self.current_application

    def apply_burn_rate(self) -> None:
        """
        Применяем burn_rate и productivity к каждой живой компании
        """
        for company in self.companies.values():
            if company.is_defaulted:
                continue
            leverage = company.bank_debt / max(1.0, company.assets)
            leverage_burn = company.burn_rate * (1.0 + leverage * 2.0)
            company.assets -= company.assets * leverage_burn
            company.assets += company.assets * (company.productivity - 1.0) * 0.1

    def execute_approve(self) -> float:
        """
        Одобрить заявку:
        1. компания получает часть суммы как реальные активы
        2. полная сумма записывается как bank_debt.
        3. часть кредита увеличивает total_liabilities, увеличивая уязвимость
        4. теперь каждый шаг компания платит (principal + interest) / term
        ret:
        0, так как доходы считаются потом
        """
        app = self.current_application
        company = self.companies[app.applicant_id]

        if company.is_defaulted:
            return 0.0

        disbursement_ratio = 0.35
        company.assets += app.amount * disbursement_ratio
        company.bank_debt += app.amount

        liability_increase = app.amount * self.config.loan_liability_passthrough
        company.total_liabilities += liability_increase
        total_payment = app.amount + app.expected_return
        payment_per_step = total_payment / app.term
        company.active_loans.append({
            "remaining_term": app.term,
            "payment_per_step": payment_per_step,
            "principal_per_step": app.amount / app.term,
        })

        self.total_approvals += 1
        self.sector_approvals[company.sector] += 1
        self.borrowers.add(company.id)
        self.total_loans_issued += app.amount

        return 0.0

    def execute_reject(self) -> float:
        """
        Отклонить заявку
        """
        self.total_rejections += 1
        return 0.0

    def process_loans(self) -> float:
        """
        Собрать платежи по активным кредитам со всех компаний:
        живая платит payment_per_step, а если активов меньше, то платит минимум
        ret:
            income: доход за шаг
        """
        income = 0.0
        for company in self.companies.values():
            if company.is_defaulted or not company.active_loans:
                continue

            remaining_loans = []
            for loan in company.active_loans:
                payment = loan["payment_per_step"]
                principal = loan["principal_per_step"]

                actual_payment = min(payment, company.assets)
                company.assets -= actual_payment
                pay_ratio = actual_payment / max(1e-8, payment)
                actual_principal = principal * pay_ratio
                company.bank_debt -= min(actual_principal, company.bank_debt)

                interest_income = (payment - principal) * pay_ratio
                income += interest_income
                self.sector_pnl[company.sector] += interest_income

                loan["remaining_term"] -= 1
                if loan["remaining_term"] > 0:
                    remaining_loans.append(loan)

            company.active_loans = remaining_loans

        return income

    def apply_shock(self) -> None:
        """
        Шокируем среду:
        1. 1 случайная компания теряет shock_asset_loss
        2. все компании случайного сектора теряют sector_shock_asset_loss
        3.  все живые компании теряют macro_shock_asset_loss
        """
        alive = [c for c in self.companies.values() if not c.is_defaulted]
        if not alive:
            return
        # solo
        if self.config.shock_probability > 0:
            if self.rng.random() < self.config.shock_probability:
                target = self.rng.choice(alive)
                target.assets *= (1.0 - self.config.shock_asset_loss)
        # sector
        if self.config.sector_shock_probability > 0:
            if self.rng.random() < self.config.sector_shock_probability:
                sector = int(self.rng.integers(0, self.config.num_sectors))
                for c in alive:
                    if c.sector == sector:
                        c.assets *= (1.0 - self.config.sector_shock_asset_loss)

        # macro
        if self.config.macro_shock_probability > 0:
            if self.rng.random() < self.config.macro_shock_probability:
                for c in alive:
                    c.assets *= (1.0 - self.config.macro_shock_asset_loss)

    def run_contagion_loop(self) -> list[int]:
        """
        Каскадное распространение дефолтов через bfs, то есть
        пока дефолты / компании не кончились:
            1. Убиваем все компании с equity < 0
            2. Партнеры компании, которая дефолтнулась,
            теряют contagion_fraction * debt_amount от активов
        return:
            new_defaults: список дефолтнутых компаний
        """
        new_defaults = []
        queue = deque()  # (company_id, cascade_depth)
        cascade_depth = 0

        for company in self.companies.values():
            if not company.is_defaulted and company.equity() < 0:
                company.is_defaulted = True
                new_defaults.append(company.id)
                queue.append((company.id, 0))

        while queue:
            defaulted_id, depth = queue.popleft()
            cascade_depth = max(cascade_depth, depth)
            defaulter = self.companies[defaulted_id]

            amp = self.config.contagion_leverage_amplifier
            leverage_mult = 1.0 + amp * defaulter.bank_debt / max(
                1.0, abs(defaulter.assets) + defaulter.bank_debt
            )

            for edge in self.edges:
                if edge.target_id == defaulted_id:
                    creditor = self.companies[edge.source_id]
                    if not creditor.is_defaulted:
                        loss = edge.debt_amount * self.config.contagion_fraction * leverage_mult
                        creditor.assets -= loss

                        if creditor.equity() < 0:
                            creditor.is_defaulted = True
                            new_defaults.append(creditor.id)
                            queue.append((creditor.id, depth + 1))

        self.total_defaults += len(new_defaults)
        self.max_cascade_depth = max(self.max_cascade_depth, cascade_depth)
        return new_defaults

    def calculate_recovery(self, defaulted_ids: list[int]) -> float:
        """
        Подсчитываем сколько денег возвращается от дефолтов,
        при банкротстве банк возвращает liquidation_cost * остаточные_активы, а default_cost отражает потери
        Returns:
            loss: убыток от дефолтов
        """
        total_loss = 0.0
        for cid in defaulted_ids:
            company = self.companies[cid]
            recovery = max(0.0, company.assets * company.liquidation_cost)
            debt_loss = max(0.0, company.bank_debt - recovery)
            fixed_cost = self.config.default_cost if company.bank_debt > 0 else 0.0
            total_loss -= (debt_loss + fixed_cost)
            self.sector_pnl[company.sector] += -(debt_loss + fixed_cost)
        return total_loss

    def process_step(self, action: int) -> float:
        """
        Шаг симуляции
        0 = reject, 1 = approve
        Returns:
            step_pnl: P&L за 1 шаг
        """
        self.step_pnl = 0.0
        self.apply_burn_rate()
        if action == 1:
            self.execute_approve()
        else:
            self.execute_reject()
            # штрафуем за невыдачу
            penalty = self.config.rejection_penalty
            if penalty > 0 and self.current_application is not None:
                self.step_pnl -= self.current_application.expected_return * penalty

        loan_income = self.process_loans()
        self.step_pnl += loan_income
        self.apply_shock()
        new_defaults = self.run_contagion_loop()
        recovery_pnl = self.calculate_recovery(new_defaults)
        self.step_pnl += recovery_pnl

        self.cumulative_pnl += self.step_pnl
        self.step_pnl_history.append(self.step_pnl)
        self.current_step += 1

        return self.step_pnl

    def run_cooldown(self) -> float:
        """
        Заканчиваем выдачу кредитов, ждем пока оставшиеся должники вернут или дефолтнут
        Returns:
            cooldown_pnl: суммарный PnL за cooldown
        """
        max_steps = getattr(self.config, "cooldown_max_steps", 30)
        cooldown_pnl = 0.0

        for i in range(max_steps):
            has_active_loans = any(
                c.active_loans
                for c in self.companies.values()
                if not c.is_defaulted
            )
            if not has_active_loans:
                break

            self.step_pnl = 0.0

            self.apply_burn_rate()

            loan_income = self.process_loans()
            self.step_pnl += loan_income

            new_defaults = self.run_contagion_loop()
            recovery = self.calculate_recovery(new_defaults)
            self.step_pnl += recovery

            cooldown_pnl += self.step_pnl
            self.cumulative_pnl += self.step_pnl
            self.step_pnl_history.append(self.step_pnl)

        return cooldown_pnl

    def get_alive_count(self) -> int:
        """
        Количество живых компаний
        """
        return sum(1 for c in self.companies.values() if not c.is_defaulted)

    def get_borrower_default_rate(self) -> float:
        """
        Доля дефолтов среди компаний, которым банк выдал кредит
        """
        if not self.borrowers:
            return 0.0
        defaulted_borrowers = sum(
            1 for cid in self.borrowers if self.companies[cid].is_defaulted
        )
        return defaulted_borrowers / len(self.borrowers)

    def get_sharpe_ratio(self) -> float:
        """
        классический Sharpe Ratio = mean(step_pnl) / std(step_pnl)
        """
        if len(self.step_pnl_history) < 2:
            return 0.0
        std = np.std(self.step_pnl_history)
        if std < 1e-8:
            return 0.0
        return float(np.mean(self.step_pnl_history) / std)

    def get_roa(self) -> float:
        """
        ROA = cumulative_pnl / total_loans_issued
        Доход на единицу выданного кредита
        """
        if self.total_loans_issued < 1e-8:
            return 0.0
        return self.cumulative_pnl / self.total_loans_issued

    def get_survival_rate(self) -> float:
        """
        Доля выживших компаний
        """
        return self.get_alive_count() / max(1, len(self.companies))

    def get_pnl_per_step(self) -> float:
        """
        Средний P&L за шаг. Нормализует на длину эпизода
        """
        if self.current_step == 0:
            return 0.0
        return self.cumulative_pnl / self.current_step

    def get_pnl_per_company(self) -> float:
        """
        P&L на компанию (со всего графа)
        """
        n = len(self.companies)
        if n == 0:
            return 0.0
        return self.cumulative_pnl / n

    def get_metrics(self) -> dict:
        """
        Собираем метрики для логов
        """
        return {
            "cumulative_pnl": self.cumulative_pnl,
            "step_pnl": self.step_pnl,
            "alive_companies": self.get_alive_count(),
            "total_defaults": self.total_defaults,
            "total_approvals": self.total_approvals,
            "total_rejections": self.total_rejections,
            "approval_rate": (
                    self.total_approvals / max(1, self.total_approvals + self.total_rejections)
            ),
            "borrower_default_rate": self.get_borrower_default_rate(),
            "num_borrowers": len(self.borrowers),
            "current_step": self.current_step,

            "sharpe_ratio": self.get_sharpe_ratio(),
            "roa": self.get_roa(),
            "survival_rate": self.get_survival_rate(),
            "max_cascade_depth": self.max_cascade_depth,
            "total_loans_issued": self.total_loans_issued,

            "pnl_per_step": self.get_pnl_per_step(),
            "pnl_per_company": self.get_pnl_per_company(),
        }
