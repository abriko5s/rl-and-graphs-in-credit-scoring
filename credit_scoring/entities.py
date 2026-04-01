"""
All essential entities: companies, edges, applications
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CompanyNode:
    """
    Компания в среде
    Public:
        id: айдишник
        is_defaulted: банкрот ли?
        sector: номер сектора
        assets: активы
        total_liabilities: все обязательства

    Private:
        bank_debt: текущий долг
        burn_rate: скорость трат активов за шаг
        productivity: процент доходов
        liquidation_cost: доля активов, возвращаемая банку при банкротстве
    """
    id: int
    sector: int
    assets: float
    total_liabilities: float
    is_defaulted: bool = False
    bank_debt: float = 0.0
    burn_rate: float = 0.02
    productivity: float = 1.0
    liquidation_cost: float = 0.4

    # Текущие кредиты (remaining_term, payment_per_step)
    active_loans: list = field(default_factory=list)

    def equity(self) -> float:
        return self.assets - self.total_liabilities - self.bank_debt


@dataclass
class SupplyEdge:
    """
    Ребро графа. Отображает то или иное обязательство компании
    Public:
        source_id: id компании-кредитора (например поставщика)
        target_id: id компании-должника (например покупателя)
        debt_amount: размер обязательства (сумма, которую target должен source.
        При банкротстве target, source теряет часть этой суммы)
    """
    source_id: int
    target_id: int
    debt_amount: float


@dataclass
class LoanApplication:
    """
    Кредитная заявка, поступающая на рассмотрение агенту

    Public:
        applicant_id: id компании
        amount: запрашиваемая сумма кредита
        term: срок кредита (в шагах симуляции)
        expected_return: ожидаемый процентный доход для банка за весь срок
    """
    applicant_id: int
    amount: float
    term: int
    expected_return: float
