from credit_scoring.entities import CompanyNode, LoanApplication, SupplyEdge


def test_company_equity_positive():
    c = CompanyNode(id=0, sector=1, assets=100.0, total_liabilities=30.0, bank_debt=20.0)
    assert c.equity() == 50.0


def test_company_equity_negative():
    c = CompanyNode(id=0, sector=1, assets=10.0, total_liabilities=30.0, bank_debt=20.0)
    assert c.equity() == -40.0


def test_company_defaults():
    c = CompanyNode(id=0, sector=0, assets=100.0, total_liabilities=50.0)
    assert not c.is_defaulted
    c.is_defaulted = True
    assert c.is_defaulted


def test_loan_application():
    app = LoanApplication(applicant_id=5, amount=50.0, term=10, expected_return=25.0)
    assert app.applicant_id == 5
    assert app.amount == 50.0
    assert app.term == 10


def test_supply_edge():
    edge = SupplyEdge(source_id=1, target_id=2, debt_amount=15.0)
    assert edge.source_id == 1
    assert edge.target_id == 2
    assert edge.debt_amount == 15.0
