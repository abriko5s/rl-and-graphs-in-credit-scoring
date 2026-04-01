import numpy as np

from credit_scoring.config import Config
from credit_scoring.graph_gen import generate_economy
from credit_scoring.market import MarketSimulator


def _make_sim(seed: int = 42, **config_kwargs) -> MarketSimulator:
    config = Config(num_companies=10, num_steps=50, **config_kwargs)
    companies, graph, edges = generate_economy(config, seed=seed)
    return MarketSimulator(
        companies=companies, graph=graph, edges=edges,
        config=config, rng=np.random.default_rng(seed),
    )


def test_sim_init():
    sim = _make_sim()
    assert len(sim.companies) == 10
    assert sim.current_step == 0
    assert sim.cumulative_pnl == 0.0


def test_generate_application():
    sim = _make_sim()
    app = sim.generate_application()
    assert app.applicant_id in sim.companies
    assert app.amount > 0
    assert app.term > 0


def test_process_step_approve():
    sim = _make_sim()
    sim.generate_application()
    pnl = sim.process_step(action=1)
    assert sim.current_step == 1
    assert sim.total_approvals == 1
    assert isinstance(pnl, float)


def test_process_step_reject():
    sim = _make_sim()
    sim.generate_application()
    pnl = sim.process_step(action=0)
    assert sim.current_step == 1
    assert sim.total_rejections == 1


def test_burn_rate_reduces_assets():
    sim = _make_sim()
    initial_assets = {c.id: c.assets for c in sim.companies.values()}
    sim.apply_burn_rate()
    # Активы должны измениться (burn - productivity income)
    for c in sim.companies.values():
        assert c.assets != initial_assets[c.id]


def test_approve_increases_assets():
    sim = _make_sim()
    app = sim.generate_application()
    company = sim.companies[app.applicant_id]
    initial_assets = company.assets
    sim.execute_approve()
    # execute_approve добавляет только app.amount * disbursement_ratio (0.35) к активам
    disbursement_ratio = 0.35
    assert abs(company.assets - (initial_assets + app.amount * disbursement_ratio)) < 1e-6
    assert company.bank_debt > 0


def test_contagion_cascades():
    """Проверить, что дефолт вызывает каскад."""
    sim = _make_sim(seed=123)
    # Банкротим одну компанию вручную
    victim = list(sim.companies.values())[0]
    victim.assets = 0.0
    victim.total_liabilities = 1000.0
    victim.bank_debt = 500.0

    defaults = sim.run_contagion_loop()
    assert len(defaults) >= 1
    assert victim.is_defaulted


def test_reset():
    sim = _make_sim()
    sim.generate_application()
    sim.process_step(action=1)
    sim.process_step(action=1)

    sim.reset()
    assert sim.current_step == 0
    assert sim.cumulative_pnl == 0.0
    assert sim.total_approvals == 0


def test_get_metrics():
    sim = _make_sim()
    sim.generate_application()
    sim.process_step(action=1)
    metrics = sim.get_metrics()
    assert "cumulative_pnl" in metrics
    assert "alive_companies" in metrics
    assert "approval_rate" in metrics


def test_full_episode():
    sim = _make_sim()
    for _ in range(50):
        sim.generate_application()
        action = 1 if np.random.random() > 0.5 else 0
        sim.process_step(action)

    assert sim.current_step == 50
    assert sim.total_approvals + sim.total_rejections <= 50
