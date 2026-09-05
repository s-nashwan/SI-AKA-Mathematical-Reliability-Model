import numpy as np
import pytest
from src.siaka_models import (
    iid_failure_probability, expected_extra_hashes, pareto_frontier,
    system_failure_probability, ge_transition_from_eta_kappa,
    ge_stationary_distribution, ge_consecutive_loss_probability,
    ge_spectral_radius, active_suppression_initial_value,
)


def test_iid_failure_strictly_decreases_with_window():
    for p in (0.03,0.1,0.25,0.5):
        for r in (0.0,0.05,0.3):
            vals=[iid_failure_probability(p,r,m) for m in range(1,10)]
            assert all(vals[i+1] < vals[i] for i in range(len(vals)-1))


def test_iid_failure_increases_with_response_loss():
    p=.14; mu=4
    vals=[iid_failure_probability(p,r,mu) for r in (0,.05,.1,.2,.4)]
    assert all(vals[i+1] > vals[i] for i in range(len(vals)-1))


def test_expected_hash_cost_converges_to_closed_limit():
    p,r=.2,.1
    limit=p/((1-p)*(1-r))
    assert expected_extra_hashes(p,r,80) == pytest.approx(limit, rel=1e-12)


def test_analytic_frontier_equals_bruteforce_pareto_set():
    pC,rC,pS,rS,eps=.08,.04,.12,.06,1e-4
    analytic={(a,b) for a,b,_ in pareto_frontier(pC,rC,pS,rS,eps,mu_c_max=12,mu_s_max=12)}
    feasible=[]
    for a in range(1,13):
        for b in range(1,13):
            f=system_failure_probability(iid_failure_probability(pC,rC,a),iid_failure_probability(pS,rS,b))
            if f<=eps:
                feasible.append((a,b))
    brute=set()
    for a,b in feasible:
        dominated=any((x<=a and y<=b and (x<a or y<b)) for x,y in feasible)
        if not dominated:
            brute.add((a,b))
    assert analytic==brute


def test_ge_consecutive_loss_decay_ratio_tracks_spectral_radius_over_grid():
    eta=.2;pG=.02;pB=.42
    for kappa in (.2,.5,.75,.9):
        T=ge_transition_from_eta_kappa(eta,kappa)
        pi=ge_stationary_distribution(T)
        rho=ge_spectral_radius(T,pG,pB)
        a=ge_consecutive_loss_probability(pi,T,pG,pB,24)
        b=ge_consecutive_loss_probability(pi,T,pG,pB,25)
        assert b/a == pytest.approx(rho, rel=2e-5)


def test_robust_failure_nondecreasing_in_budget_over_grid():
    T=ge_transition_from_eta_kappa(.2,.8)
    vals=[active_suppression_initial_value(T,.02,.42,.05,7,horizon=18,budget=b) for b in range(0,6)]
    assert all(vals[i+1]+1e-14 >= vals[i] for i in range(len(vals)-1))
