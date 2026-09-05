import math
import numpy as np
import pytest

from src.siaka_models import (
    iid_failure_probability,
    iid_min_window,
    delivered_request_mean,
    cycle_request_moments,
    total_request_moments,
    conditional_first_passage_means,
    request_pgf,
    system_failure_probability,
    sensor_failure_budget,
    pareto_frontier,
    ge_transition_from_eta_kappa,
    ge_stationary_distribution,
    ge_mean_loss,
    ge_loss_autocorrelation,
    ge_consecutive_loss_probability,
    ge_spectral_radius,
    build_ge_absorbing_chain,
    ge_absorption_probabilities,
    active_suppression_value,
)


def test_iid_failure_matches_closed_form_reference_case():
    p, r, mu = 0.10, 0.05, 3
    expected = p**mu / (p**mu + (1-p**mu)*(1-r))
    assert iid_failure_probability(p, r, mu) == pytest.approx(expected, rel=1e-14)


def test_min_window_is_minimal():
    p, r, eps = 0.12, 0.06, 1e-4
    mu = iid_min_window(p, r, eps)
    assert iid_failure_probability(p, r, mu) <= eps
    assert iid_failure_probability(p, r, mu-1) > eps


def test_delivered_request_mean_matches_direct_sum():
    p, mu = 0.17, 6
    q = 1-p
    direct = sum(k*q*p**(k-1) for k in range(1, mu+1)) / (1-p**mu)
    assert delivered_request_mean(p, mu) == pytest.approx(direct, rel=1e-13)


def test_cycle_request_moments_match_direct_distribution():
    p, mu = 0.23, 5
    probs = [((1-p)*p**(k-1)) for k in range(1, mu)] + [p**(mu-1)]
    vals = np.arange(1, mu+1)
    m1 = float(np.dot(vals, probs))
    m2 = float(np.dot(vals**2, probs))
    got1, got2 = cycle_request_moments(p, mu)
    assert got1 == pytest.approx(m1, rel=1e-13)
    assert got2 == pytest.approx(m2, rel=1e-13)


def test_pgf_normalizes_and_derivative_matches_total_mean():
    p, r, mu = 0.14, 0.09, 4
    assert request_pgf(1.0, p, r, mu) == pytest.approx(1.0, abs=1e-13)
    h = 1e-6
    deriv = (request_pgf(1+h,p,r,mu)-request_pgf(1-h,p,r,mu))/(2*h)
    mean, var = total_request_moments(p, r, mu)
    assert deriv == pytest.approx(mean, rel=2e-6)
    assert var > 0


def test_conditional_first_passage_means_match_exact_enumeration_logic():
    p, r, mu = 0.2, 0.1, 4
    success, boundary, rejection = conditional_first_passage_means(p, r, mu)
    mD = delivered_request_mean(p, mu)
    Y = (1-p**mu)*r
    d = 1-Y
    assert success == pytest.approx(mD/d)
    assert boundary == pytest.approx(mu + (Y/d)*mD)
    assert rejection == pytest.approx(boundary + 1/(1-p))


def test_system_frontier_is_upward_closed_and_pareto_minimal():
    pC,rC,pS,rS,eps = 0.08,0.04,0.12,0.06,1e-4
    frontier = pareto_frontier(pC,rC,pS,rS,eps,mu_c_max=12,mu_s_max=12)
    assert frontier
    for muC, muS, fsys in frontier:
        assert fsys <= eps
        if muC > 1:
            assert system_failure_probability(
                iid_failure_probability(pC,rC,muC-1),
                iid_failure_probability(pS,rS,muS)
            ) > eps or (muC-1,muS) not in [(x,y) for x,y,_ in frontier]
        if muS > 1:
            assert system_failure_probability(
                iid_failure_probability(pC,rC,muC),
                iid_failure_probability(pS,rS,muS-1)
            ) > eps


def test_sensor_failure_budget_formula_exact():
    Fc, eps = 4e-5, 1e-4
    budget = sensor_failure_budget(Fc, eps)
    assert 1-(1-Fc)*(1-budget) == pytest.approx(eps, rel=1e-14)


def test_ge_parameterization_has_requested_stationary_distribution_and_persistence():
    eta, kappa = 0.2, 0.75
    T = ge_transition_from_eta_kappa(eta,kappa)
    pi = ge_stationary_distribution(T)
    assert pi == pytest.approx(np.array([0.8,0.2]), abs=1e-13)
    eig = sorted(np.linalg.eigvals(T), key=lambda x: abs(x), reverse=True)
    assert eig[1].real == pytest.approx(kappa, rel=1e-13)


def test_ge_autocorrelation_matches_monte_carlo_formula_components():
    eta,kappa,pG,pB = 0.25,0.7,0.02,0.34
    pbar = ge_mean_loss(eta,pG,pB)
    rho1 = ge_loss_autocorrelation(eta,kappa,pG,pB,1)
    cov1 = eta*(1-eta)*(pB-pG)**2*kappa
    assert rho1 == pytest.approx(cov1/(pbar*(1-pbar)), rel=1e-13)


def test_ge_reduces_to_iid_for_equal_emissions():
    eta,kappa,p,mu = 0.35,0.82,0.1,6
    T = ge_transition_from_eta_kappa(eta,kappa)
    pi = ge_stationary_distribution(T)
    b = ge_consecutive_loss_probability(pi,T,p,p,mu)
    assert b == pytest.approx(p**mu, rel=2e-13)
    assert ge_spectral_radius(T,p,p) == pytest.approx(p, rel=2e-13)


def test_ge_spectral_ratio_converges_to_dominant_radius():
    eta,kappa,pG,pB = 0.2,0.85,0.01,0.46
    T = ge_transition_from_eta_kappa(eta,kappa)
    pi = ge_stationary_distribution(T)
    sr = ge_spectral_radius(T,pG,pB)
    vals = [ge_consecutive_loss_probability(pi,T,pG,pB,m) for m in range(18,23)]
    ratios = [vals[i+1]/vals[i] for i in range(len(vals)-1)]
    assert ratios[-1] == pytest.approx(sr, rel=2e-4)


def test_ge_absorbing_chain_probabilities_sum_to_one_and_reduce_to_iid():
    p,r,mu = 0.1,0.05,5
    eta,kappa = 0.3,0.7
    T = ge_transition_from_eta_kappa(eta,kappa)
    pi = ge_stationary_distribution(T)
    Q,bf,bs = build_ge_absorbing_chain(T,p,p,r,mu)
    fail,succ = ge_absorption_probabilities(Q,bf,bs,pi)
    assert fail+succ == pytest.approx(1.0, abs=2e-13)
    assert fail == pytest.approx(iid_failure_probability(p,r,mu), rel=2e-12)


def test_active_suppression_monotone_in_budget_gap_horizon():
    eta,kappa,pG,pB,r,mu = 0.25,0.75,0.02,0.35,0.05,5
    T = ge_transition_from_eta_kappa(eta,kappa)
    # bad-state index = 1
    base = active_suppression_value(T,pG,pB,r,mu,horizon=8,gap=1,state=1,budget=1)
    more_b = active_suppression_value(T,pG,pB,r,mu,horizon=8,gap=1,state=1,budget=2)
    more_d = active_suppression_value(T,pG,pB,r,mu,horizon=8,gap=2,state=1,budget=1)
    more_h = active_suppression_value(T,pG,pB,r,mu,horizon=9,gap=1,state=1,budget=1)
    assert more_b + 1e-14 >= base
    assert more_d + 1e-14 >= base
    assert more_h + 1e-14 >= base


def test_suppression_can_force_boundary_when_budget_and_horizon_cover_window():
    T = ge_transition_from_eta_kappa(0.2,0.6)
    for state in (0,1):
        v = active_suppression_value(T,0.01,0.2,0.05,mu=4,horizon=4,gap=0,state=state,budget=4)
        assert v == pytest.approx(1.0, abs=1e-15)

from src.siaka_models import (
    ap_reachable_states,
    iid_monte_carlo,
    wilson_interval,
    ge_monte_carlo,
    ge_spectral_radius_closed_form,
)


def test_ap_exhaustive_reachable_hash_gap_is_at_most_one():
    states = ap_reachable_states(max_depth=12)
    assert states
    gaps = {s[1] for s in states}
    assert gaps == {0,1}
    assert max(gaps) == 1


def test_total_variance_matches_pgf_second_derivative():
    p,r,mu = 0.19,0.08,5
    mean,var = total_request_moments(p,r,mu)
    h=2e-4
    g0=request_pgf(1.0,p,r,mu)
    gp=request_pgf(1.0+h,p,r,mu)
    gm=request_pgf(1.0-h,p,r,mu)
    second=(gp-2*g0+gm)/(h*h)
    # For a PGF, E[T^2]=G''(1)+G'(1).
    pgf_var=second+mean-mean**2
    assert pgf_var == pytest.approx(var, rel=2e-5)


def test_iid_monte_carlo_contains_analytic_failure_in_wilson_interval():
    p,r,mu=0.1,0.05,3
    sim=iid_monte_carlo(p,r,mu,trials=250_000,seed=20260905)
    lo,hi=wilson_interval(sim['failures'],sim['trials'])
    ana=iid_failure_probability(p,r,mu)
    assert lo <= ana <= hi
    assert sim['mean_requests'] == pytest.approx(total_request_moments(p,r,mu)[0], rel=0.01)


def test_ge_closed_form_spectral_radius_matches_matrix_eigenvalue():
    eta,kappa,pG,pB=.2,.83,.015,.44
    T=ge_transition_from_eta_kappa(eta,kappa)
    assert ge_spectral_radius_closed_form(eta,kappa,pG,pB) == pytest.approx(
        ge_spectral_radius(T,pG,pB), rel=2e-13
    )


def test_ge_monte_carlo_matches_absorbing_markov_probability():
    eta,kappa,pG,pB,r,mu=.2,.7,.02,.42,.05,6
    T=ge_transition_from_eta_kappa(eta,kappa)
    pi=ge_stationary_distribution(T)
    Q,bf,bs=build_ge_absorbing_chain(T,pG,pB,r,mu)
    ana,_=ge_absorption_probabilities(Q,bf,bs,pi)
    sim=ge_monte_carlo(T,pG,pB,r,mu,trials=180_000,seed=77)
    lo,hi=wilson_interval(sim['failures'],sim['trials'])
    assert lo <= ana <= hi

from src.siaka_models import active_suppression_initial_value, robust_min_window


def test_active_initial_value_is_stationary_mixture():
    eta,kappa=.2,.8
    T=ge_transition_from_eta_kappa(eta,kappa)
    pi=ge_stationary_distribution(T)
    vals=[active_suppression_value(T,.02,.42,.05,5,horizon=12,gap=0,state=z,budget=2) for z in (0,1)]
    assert active_suppression_initial_value(T,.02,.42,.05,5,horizon=12,budget=2) == pytest.approx(
        float(pi @ np.array(vals)), rel=1e-13
    )


def test_robust_min_window_obeys_budget_lower_bound_and_target():
    T=ge_transition_from_eta_kappa(.2,.75)
    eps=1e-3
    mu=robust_min_window(T,.02,.42,.05,horizon=16,budget=3,epsilon=eps,mu_max=20)
    assert mu >= 4
    assert active_suppression_initial_value(T,.02,.42,.05,mu,horizon=16,budget=3) <= eps
    if mu>1:
        assert active_suppression_initial_value(T,.02,.42,.05,mu-1,horizon=16,budget=3) > eps

from src.siaka_models import request_pmf


def test_request_pmf_matches_pgf_on_unit_interval_and_moments():
    p,r,mu=.18,.12,4
    pmf=request_pmf(p,r,mu,t_max=120)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-11)
    t=np.arange(len(pmf))
    mean=float(np.dot(t,pmf))
    var=float(np.dot((t-mean)**2,pmf))
    m,v=total_request_moments(p,r,mu)
    assert mean == pytest.approx(m, rel=1e-10)
    assert var == pytest.approx(v, rel=1e-9)
    z=.87
    assert float(np.dot(pmf,z**t)) == pytest.approx(request_pgf(z,p,r,mu), rel=1e-10)

from src.siaka_models import robust_system_frontier


def test_robust_system_frontier_returns_nondominated_feasible_pairs():
    TC=ge_transition_from_eta_kappa(.15,.7)
    TS=ge_transition_from_eta_kappa(.2,.8)
    eps=2e-3
    fr=robust_system_frontier(
        TC,.015,.30,.04,14,1,
        TS,.02,.42,.05,14,2,
        epsilon_sys=eps,mu_c_max=14,mu_s_max=16
    )
    assert fr
    pairs=[(a,b) for a,b,_ in fr]
    for a,b,f in fr:
        assert f<=eps
        assert not any(x<=a and y<=b and (x<a or y<b) for x,y in pairs)

from src.siaka_models import (
    ge_cycle_kernel,
    ge_matrix_renewal_metrics,
    ge_expected_requests,
    ge_expected_extra_hashes,
    expected_extra_hashes,
)


def test_ge_matrix_renewal_matches_absorbing_chain_failure_and_metrics():
    eta,kappa,pG,pB,r,mu=.2,.82,.02,.42,.05,7
    T=ge_transition_from_eta_kappa(eta,kappa)
    pi=ge_stationary_distribution(T)
    Q,bf,bs=build_ge_absorbing_chain(T,pG,pB,r,mu)
    f_abs,s_abs=ge_absorption_probabilities(Q,bf,bs,pi)
    t_abs=ge_expected_requests(Q,pi)
    h_abs=ge_expected_extra_hashes(Q,pi,pG,pB,mu)
    metrics=ge_matrix_renewal_metrics(T,pG,pB,r,mu,pi=pi)
    assert metrics['failure_probability'] == pytest.approx(f_abs, rel=2e-12)
    assert metrics['success_probability'] == pytest.approx(s_abs, rel=2e-12)
    assert metrics['mean_requests'] == pytest.approx(t_abs, rel=2e-12)
    assert metrics['expected_extra_hashes'] == pytest.approx(h_abs, rel=2e-12)


def test_ge_cycle_kernel_probability_balance_from_each_start_state():
    T=ge_transition_from_eta_kappa(.3,.7)
    mu=5
    K,b=ge_cycle_kernel(T,.03,.31,mu)
    assert K.shape == (2,2)
    assert b.shape == (2,)
    assert np.allclose(K @ np.ones(2) + b, np.ones(2), atol=2e-13)


def test_ge_matrix_renewal_reduces_to_iid_exactly_for_equal_emissions():
    p,r,mu=.11,.07,6
    T=ge_transition_from_eta_kappa(.25,.88)
    pi=ge_stationary_distribution(T)
    metrics=ge_matrix_renewal_metrics(T,p,p,r,mu,pi=pi)
    m_iid,v_iid=total_request_moments(p,r,mu)
    assert metrics['failure_probability'] == pytest.approx(iid_failure_probability(p,r,mu), rel=2e-12)
    assert metrics['mean_requests'] == pytest.approx(m_iid, rel=2e-12)
    assert metrics['expected_extra_hashes'] == pytest.approx(expected_extra_hashes(p,r,mu), rel=2e-12)


def test_robust_min_window_uses_horizon_limited_deterministic_lower_bound():
    T=ge_transition_from_eta_kappa(.2,.6)
    # With only H=2 opportunities, any mu<=2 can be forced when budget>=2;
    # the first potentially non-deterministic window is therefore H+1=3.
    mu=robust_min_window(T,.0,.0,.0,horizon=2,budget=5,epsilon=.5,mu_max=8)
    assert mu >= 3
