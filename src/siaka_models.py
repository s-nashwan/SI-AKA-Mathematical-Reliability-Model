"""Mathematical models for SI-AKA synchronization analysis.

The module models only the abstract synchronization rules supported by the
published SI-AKA protocol: a bounded AP--AuC recovery state and two explicit
window-based request/response links.  No cryptographic message is modified.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterable

import numpy as np


def _check_prob(x: float, name: str = "probability", *, strict_one: bool = False) -> None:
    upper_ok = x < 1.0 if strict_one else x <= 1.0
    if x < 0.0 or not upper_ok:
        op = "[0,1)" if strict_one else "[0,1]"
        raise ValueError(f"{name} must lie in {op}")


def _check_mu(mu: int) -> None:
    if int(mu) != mu or mu < 1:
        raise ValueError("mu must be a positive integer")


def iid_failure_probability(p: float, r: float, mu: int) -> float:
    """Eventual loss-of-recoverability probability for one IID window link."""
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    x = p ** mu
    d = x + (1.0 - x) * (1.0 - r)
    return x / d


def iid_success_probability(p: float, r: float, mu: int) -> float:
    return 1.0 - iid_failure_probability(p, r, mu)


def iid_min_window(p: float, r: float, epsilon: float) -> int:
    """Smallest integer mu such that the IID failure probability <= epsilon."""
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0,1)")
    if p == 0.0:
        return 1
    threshold = epsilon * (1.0 - r) / (1.0 - epsilon * r)
    # p^mu <= threshold; log(p)<0 reverses the real-valued inequality.
    real_mu = math.log(threshold) / math.log(p)
    mu = max(1, math.ceil(real_mu - 1e-14))
    # Protect against floating-point boundary cases.
    while iid_failure_probability(p, r, mu) > epsilon:
        mu += 1
    while mu > 1 and iid_failure_probability(p, r, mu - 1) <= epsilon:
        mu -= 1
    return mu


def delivered_request_mean(p: float, mu: int) -> float:
    """E[K | request delivered before boundary], K in {1,...,mu}."""
    _check_prob(p, "p", strict_one=True)
    _check_mu(mu)
    if p == 0.0:
        return 1.0
    x = p ** mu
    q = 1.0 - p
    num = 1.0 - (mu + 1.0) * x + mu * x * p
    return num / (q * (1.0 - x))


def cycle_request_moments(p: float, mu: int) -> tuple[float, float]:
    """First and second raw moments of C=min(G,mu), G~Geom(1-p)."""
    _check_prob(p, "p", strict_one=True)
    _check_mu(mu)
    q = 1.0 - p
    x = p ** mu
    m1 = (1.0 - x) / q
    # Tail-sum identity: E[C^2]=sum_{k=1}^mu (2k-1) p^(k-1).
    m2 = (
        1.0
        + p
        - (2.0 * mu + 1.0) * x
        + (2.0 * mu - 1.0) * x * p
    ) / (q * q)
    return m1, m2


def _delivered_first_raw_unnormalized(p: float, mu: int) -> float:
    """E[C * 1{delivery within window}] in one cycle."""
    q = 1.0 - p
    x = p ** mu
    return (1.0 - (mu + 1.0) * x + mu * x * p) / q


def total_request_moments(p: float, r: float, mu: int) -> tuple[float, float]:
    """Mean and variance of total requests until success or boundary.

    A delivered request followed by a lost response renews the cycle.
    """
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    c1, c2 = cycle_request_moments(p, mu)
    x = p ** mu
    y = (1.0 - x) * r
    d = 1.0 - y
    mean = c1 / d
    e_c_retry = r * _delivered_first_raw_unnormalized(p, mu)
    second = (c2 + 2.0 * e_c_retry * mean) / d
    var = max(0.0, second - mean * mean)
    return mean, var


def request_pgf(z: complex, p: float, r: float, mu: int) -> complex:
    """Probability generating function of total request count."""
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    q = 1.0 - p
    # Direct finite sum is stable and avoids a removable singularity at p*z=1.
    A = sum(q * (p ** (k - 1)) * (z ** k) for k in range(1, mu + 1))
    B = (p ** mu) * (z ** mu)
    return (B + (1.0 - r) * A) / (1.0 - r * A)


def conditional_first_passage_means(p: float, r: float, mu: int) -> tuple[float, float, float]:
    """Mean requests conditional on terminal success, boundary, and observed rejection."""
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    mD = delivered_request_mean(p, mu)
    x = p ** mu
    y = (1.0 - x) * r
    d = 1.0 - y
    success = mD / d
    boundary = mu + (y / d) * mD
    rejection = boundary + 1.0 / (1.0 - p)
    return success, boundary, rejection


def expected_response_count(p: float, r: float, mu: int) -> float:
    """Expected number of response transmissions until terminal success/boundary."""
    x = p ** mu
    d = 1.0 - (1.0 - x) * r
    return (1.0 - x) / d


def expected_extra_hashes(p: float, r: float, mu: int) -> float:
    """Expected additional catch-up hashes across all delivered retry cycles."""
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    q = 1.0 - p
    x = p ** mu
    d = 1.0 - (1.0 - x) * r
    num = p - mu * x + (mu - 1.0) * x * p
    return num / (q * d)


def system_failure_probability(Fc: float, Fs: float) -> float:
    _check_prob(Fc, "Fc")
    _check_prob(Fs, "Fs")
    return 1.0 - (1.0 - Fc) * (1.0 - Fs)


def sensor_failure_budget(Fc: float, epsilon_sys: float) -> float:
    """Largest admissible sensor-link failure given Fc and a system target."""
    _check_prob(Fc, "Fc")
    if not (0.0 < epsilon_sys < 1.0):
        raise ValueError("epsilon_sys must lie in (0,1)")
    if Fc >= epsilon_sys:
        return 0.0
    return (epsilon_sys - Fc) / (1.0 - Fc)


def pareto_frontier(
    p_c: float,
    r_c: float,
    p_s: float,
    r_s: float,
    epsilon_sys: float,
    *,
    mu_c_max: int = 30,
    mu_s_max: int = 30,
) -> list[tuple[int, int, float]]:
    """Exact Pareto-minimal staircase of feasible (mu2,mu0) pairs.

    For each mu2, the minimum compatible mu0 is computed analytically from
    the residual system failure budget. Dominated duplicate steps are removed.
    """
    out: list[tuple[int, int, float]] = []
    best_sensor = math.inf
    for mu_c in range(1, mu_c_max + 1):
        Fc = iid_failure_probability(p_c, r_c, mu_c)
        eps_s = sensor_failure_budget(Fc, epsilon_sys)
        if eps_s <= 0.0:
            continue
        mu_s = iid_min_window(p_s, r_s, eps_s)
        if mu_s > mu_s_max:
            continue
        Fs = iid_failure_probability(p_s, r_s, mu_s)
        fsys = system_failure_probability(Fc, Fs)
        if fsys <= epsilon_sys * (1.0 + 1e-12) and mu_s < best_sensor:
            out.append((mu_c, mu_s, fsys))
            best_sensor = mu_s
    return out


def ge_transition_from_eta_kappa(eta: float, kappa: float) -> np.ndarray:
    """Two-state GE transition matrix with stationary bad fraction eta and persistence kappa."""
    _check_prob(eta, "eta")
    if not (0.0 <= kappa < 1.0):
        raise ValueError("kappa must lie in [0,1)")
    a = (1.0 - kappa) * eta
    b = (1.0 - kappa) * (1.0 - eta)
    return np.array([[1.0 - a, a], [b, 1.0 - b]], dtype=float)


def ge_stationary_distribution(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (2, 2):
        raise ValueError("T must be 2x2")
    A = np.vstack([T.T - np.eye(2), np.ones(2)])
    b = np.array([0.0, 0.0, 1.0])
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.real_if_close(pi).astype(float)
    pi /= pi.sum()
    return pi


def ge_mean_loss(eta: float, p_g: float, p_b: float) -> float:
    _check_prob(eta, "eta")
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    return (1.0 - eta) * p_g + eta * p_b


def ge_loss_autocovariance(eta: float, kappa: float, p_g: float, p_b: float, lag: int) -> float:
    if lag < 1:
        raise ValueError("lag must be >=1")
    if not (0.0 <= kappa < 1.0):
        raise ValueError("kappa must lie in [0,1)")
    return eta * (1.0 - eta) * (p_b - p_g) ** 2 * (kappa ** lag)


def ge_loss_autocorrelation(eta: float, kappa: float, p_g: float, p_b: float, lag: int) -> float:
    pbar = ge_mean_loss(eta, p_g, p_b)
    if pbar in (0.0, 1.0):
        return 0.0
    return ge_loss_autocovariance(eta, kappa, p_g, p_b, lag) / (pbar * (1.0 - pbar))


def ge_consecutive_loss_probability(pi: Iterable[float], T: np.ndarray, p_g: float, p_b: float, mu: int) -> float:
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    _check_mu(mu)
    pi = np.asarray(list(pi), dtype=float)
    L = np.diag([p_g, p_b])
    A = L @ np.asarray(T, dtype=float)
    return float(pi @ np.linalg.matrix_power(A, mu) @ np.ones(2))


def ge_spectral_radius(T: np.ndarray, p_g: float, p_b: float) -> float:
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    A = np.diag([p_g, p_b]) @ np.asarray(T, dtype=float)
    return float(max(abs(np.linalg.eigvals(A))))


def ge_spectral_radius_closed_form(eta: float, kappa: float, p_g: float, p_b: float) -> float:
    T = ge_transition_from_eta_kappa(eta, kappa)
    a = T[0, 1]
    b = T[1, 0]
    tau = p_g * (1.0 - a) + p_b * (1.0 - b)
    det = p_g * p_b * kappa
    disc = max(0.0, tau * tau - 4.0 * det)
    return 0.5 * (tau + math.sqrt(disc))



def ge_cycle_kernel(
    T: np.ndarray,
    p_g: float,
    p_b: float,
    mu: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact GE one-cycle delivery kernel and boundary vector.

    Emission occurs in the current channel state and is followed by one
    channel transition.  If ``A = L @ T`` and ``S = I-L``, then

        K_mu = sum_{k=0}^{mu-1} A^k S T

    maps the channel state at the beginning of an aligned cycle to the
    channel state immediately after the first delivered request within the
    recovery window.  The vector ``b_mu = A^mu 1`` gives the probability of
    reaching the loss-of-recoverability boundary in that cycle from each
    starting channel state.
    """
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    _check_mu(mu)
    T = np.asarray(T, dtype=float)
    if T.shape != (2, 2):
        raise ValueError("T must be 2x2")
    L = np.diag([p_g, p_b])
    S = np.eye(2) - L
    A = L @ T
    K = np.zeros((2, 2), dtype=float)
    Ak = np.eye(2)
    for _ in range(mu):
        K += Ak @ S @ T
        Ak = Ak @ A
    b = Ak @ np.ones(2)
    return K, b


def ge_matrix_renewal_metrics(
    T: np.ndarray,
    p_g: float,
    p_b: float,
    r: float,
    mu: int,
    *,
    pi: Iterable[float] | None = None,
) -> dict:
    """Exact two-state matrix-renewal metrics for a complete GE link.

    This is an algebraically compressed representation of the 2*mu-state
    absorbing chain.  A delivered request followed by a lost response renews
    the aligned cycle while preserving the post-request channel state through
    the cycle kernel K_mu.
    """
    _check_prob(r, "r", strict_one=True)
    K, b = ge_cycle_kernel(T, p_g, p_b, mu)
    T = np.asarray(T, dtype=float)
    if pi is None:
        pi_vec = ge_stationary_distribution(T)
    else:
        pi_vec = np.asarray(list(pi), dtype=float)
        if pi_vec.shape != (2,):
            raise ValueError("pi must contain two probabilities")
        if np.any(pi_vec < 0.0) or not np.isclose(pi_vec.sum(), 1.0, atol=1e-12):
            raise ValueError("pi must be a probability vector")

    M = np.linalg.inv(np.eye(2) - r * K)
    one = np.ones(2)

    failure_vec = M @ b
    success_vec = M @ ((1.0 - r) * (K @ one))

    L = np.diag([p_g, p_b])
    S = np.eye(2) - L
    A = L @ T
    c = np.zeros(2, dtype=float)
    h0 = np.zeros(2, dtype=float)
    Ak = np.eye(2)
    for k in range(mu):
        c += Ak @ one
        h0 += k * (Ak @ S @ one)
        Ak = Ak @ A

    request_vec = M @ c
    response_vec = M @ (K @ one)
    hash_vec = M @ h0

    return {
        "failure_probability": float(pi_vec @ failure_vec),
        "success_probability": float(pi_vec @ success_vec),
        "mean_requests": float(pi_vec @ request_vec),
        "mean_responses": float(pi_vec @ response_vec),
        "expected_extra_hashes": float(pi_vec @ hash_vec),
        "failure_by_start_state": failure_vec,
        "success_by_start_state": success_vec,
        "mean_requests_by_start_state": request_vec,
        "cycle_kernel": K,
        "boundary_by_start_state": b,
    }


def build_ge_absorbing_chain(T: np.ndarray, p_g: float, p_b: float, r: float, mu: int):
    """Build Q and absorption vectors for GE request loss + IID response loss.

    Transient states are (d,z), d=0,...,mu-1 and z in {G,B}. Emission
    occurs in state z, followed by the channel-state transition T.
    """
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    T = np.asarray(T, dtype=float)
    n = 2 * mu
    Q = np.zeros((n, n), dtype=float)
    b_fail = np.zeros(n, dtype=float)
    b_success = np.zeros(n, dtype=float)
    pz = np.array([p_g, p_b], dtype=float)
    for d in range(mu):
        for z in range(2):
            i = 2 * d + z
            pl = pz[z]
            q = 1.0 - pl
            # Lost request.
            if d == mu - 1:
                b_fail[i] += pl
            else:
                for zp in range(2):
                    Q[i, 2 * (d + 1) + zp] += pl * T[z, zp]
            # Delivered request, lost response -> renewal at zero gap.
            for zp in range(2):
                Q[i, zp] += q * r * T[z, zp]
            # Delivered request, delivered response -> successful absorption.
            b_success[i] += q * (1.0 - r)
    return Q, b_fail, b_success


def ge_absorption_probabilities(Q: np.ndarray, b_fail: np.ndarray, b_success: np.ndarray, pi: Iterable[float]):
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if n % 2:
        raise ValueError("Q size must be even")
    alpha = np.zeros(n, dtype=float)
    pi = np.asarray(list(pi), dtype=float)
    alpha[:2] = pi
    N = np.linalg.inv(np.eye(n) - Q)
    fail = float(alpha @ N @ np.asarray(b_fail, dtype=float))
    succ = float(alpha @ N @ np.asarray(b_success, dtype=float))
    return fail, succ


def ge_expected_requests(Q: np.ndarray, pi: Iterable[float]) -> float:
    Q = np.asarray(Q, dtype=float)
    alpha = np.zeros(Q.shape[0], dtype=float)
    alpha[:2] = np.asarray(list(pi), dtype=float)
    N = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    return float(alpha @ N @ np.ones(Q.shape[0]))


def ge_expected_extra_hashes(Q: np.ndarray, pi: Iterable[float], p_g: float, p_b: float, mu: int) -> float:
    """Expected extra catch-up hashes, using reward d on delivered requests."""
    pz = np.array([p_g, p_b], dtype=float)
    g = np.zeros(2 * mu, dtype=float)
    for d in range(mu):
        for z in range(2):
            g[2 * d + z] = d * (1.0 - pz[z])
    alpha = np.zeros(2 * mu, dtype=float)
    alpha[:2] = np.asarray(list(pi), dtype=float)
    N = np.linalg.inv(np.eye(2 * mu) - Q)
    return float(alpha @ N @ g)


def active_suppression_value(
    T: np.ndarray,
    p_g: float,
    p_b: float,
    r: float,
    mu: int,
    *,
    horizon: int,
    gap: int = 0,
    state: int = 0,
    budget: int = 0,
) -> float:
    """Worst-case finite-horizon boundary probability under bounded suppression."""
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    if horizon < 0 or budget < 0 or gap < 0 or gap >= mu or state not in (0, 1):
        raise ValueError("invalid finite-horizon state")
    T = np.asarray(T, dtype=float)
    pz = (p_g, p_b)

    @lru_cache(maxsize=None)
    def V(h: int, d: int, z: int, b: int) -> float:
        if h == 0:
            return 0.0

        # Pass the request through the natural channel.
        pl = pz[z]
        if d == mu - 1:
            loss_future = 1.0
        else:
            loss_future = sum(T[z, zp] * V(h - 1, d + 1, zp, b) for zp in (0, 1))
        retry_future = sum(T[z, zp] * V(h - 1, 0, zp, b) for zp in (0, 1))
        pass_value = pl * loss_future + (1.0 - pl) * r * retry_future

        if b == 0:
            return pass_value

        # Suppression guarantees request loss and consumes one unit of budget.
        if d == mu - 1:
            suppress_value = 1.0
        else:
            suppress_value = sum(T[z, zp] * V(h - 1, d + 1, zp, b - 1) for zp in (0, 1))
        return max(pass_value, suppress_value)

    return V(horizon, gap, state, budget)


def ap_reachable_states(max_depth: int = 10) -> set[tuple[str, int]]:
    """Exhaustively enumerate the abstract AP--AuC states up to max_depth.

    The automaton encodes only the recovery transitions stated by SI-AKA:
    S --M1--> P --M4--> A --M5--> S, with M4/M5 losses waiting in
    P/A and a subsequent valid M1 repairing A to the zero-gap pending path.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    trans = {
        ("S", 0): {("S", 0), ("P", 0)},
        ("P", 0): {("P", 0), ("A", 1)},
        ("A", 1): {("A", 1), ("S", 0), ("P", 0)},
    }
    reached = {("S", 0)}
    frontier = {("S", 0)}
    for _ in range(max_depth):
        nxt = set()
        for s in frontier:
            nxt |= trans.get(s, set())
        nxt -= reached
        if not nxt:
            break
        reached |= nxt
        frontier = nxt
    return reached


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    phat = successes / trials
    denom = 1.0 + z * z / trials
    center = (phat + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def iid_monte_carlo(p: float, r: float, mu: int, *, trials: int = 100_000, seed: int = 1) -> dict:
    """Monte Carlo verification of the IID renewal model."""
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed)
    active = np.ones(trials, dtype=bool)
    failed = np.zeros(trials, dtype=bool)
    requests = np.zeros(trials, dtype=np.int64)
    q = 1.0 - p
    while np.any(active):
        idx = np.flatnonzero(active)
        # K is the request index of the first delivery in this cycle.
        K = rng.geometric(q, size=idx.size)
        used = np.minimum(K, mu)
        requests[idx] += used
        boundary = K > mu
        if np.any(boundary):
            bi = idx[boundary]
            failed[bi] = True
            active[bi] = False
        delivered_idx = idx[~boundary]
        if delivered_idx.size:
            response_lost = rng.random(delivered_idx.size) < r
            success_idx = delivered_idx[~response_lost]
            active[success_idx] = False
            # response-lost trials remain active for a new cycle
    return {
        "trials": int(trials),
        "failures": int(failed.sum()),
        "failure_probability": float(failed.mean()),
        "mean_requests": float(requests.mean()),
        "request_variance": float(requests.var(ddof=0)),
    }


def _sample_next_states(rng: np.random.Generator, current: np.ndarray, T: np.ndarray) -> np.ndarray:
    u = rng.random(current.size)
    p_to_bad = np.where(current == 0, T[0, 1], T[1, 1])
    return (u < p_to_bad).astype(np.int8)


def ge_monte_carlo(
    T: np.ndarray,
    p_g: float,
    p_b: float,
    r: float,
    mu: int,
    *,
    trials: int = 100_000,
    seed: int = 1,
) -> dict:
    """Monte Carlo verification for the two-state GE absorbing model."""
    _check_prob(p_g, "p_g")
    _check_prob(p_b, "p_b")
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    if trials <= 0:
        raise ValueError("trials must be positive")
    T = np.asarray(T, dtype=float)
    pi = ge_stationary_distribution(T)
    rng = np.random.default_rng(seed)
    state = (rng.random(trials) >= pi[0]).astype(np.int8)
    gap = np.zeros(trials, dtype=np.int16)
    requests = np.zeros(trials, dtype=np.int64)
    active = np.ones(trials, dtype=bool)
    failed = np.zeros(trials, dtype=bool)
    pz = np.array([p_g, p_b], dtype=float)

    while np.any(active):
        idx = np.flatnonzero(active)
        z = state[idx]
        requests[idx] += 1
        lost = rng.random(idx.size) < pz[z]

        # Lost requests either advance the gap or hit the boundary.
        if np.any(lost):
            li = idx[lost]
            boundary = gap[li] == mu - 1
            if np.any(boundary):
                bi = li[boundary]
                failed[bi] = True
                active[bi] = False
            cont_li = li[~boundary]
            if cont_li.size:
                gap[cont_li] += 1
                state[cont_li] = _sample_next_states(rng, state[cont_li], T)

        # Delivered requests reset the gap; a delivered response terminates.
        di = idx[~lost]
        if di.size:
            resp_lost = rng.random(di.size) < r
            success_i = di[~resp_lost]
            active[success_i] = False
            retry_i = di[resp_lost]
            if retry_i.size:
                gap[retry_i] = 0
                state[retry_i] = _sample_next_states(rng, state[retry_i], T)

    return {
        "trials": int(trials),
        "failures": int(failed.sum()),
        "failure_probability": float(failed.mean()),
        "mean_requests": float(requests.mean()),
        "request_variance": float(requests.var(ddof=0)),
    }


def active_suppression_initial_value(
    T: np.ndarray,
    p_g: float,
    p_b: float,
    r: float,
    mu: int,
    *,
    horizon: int,
    budget: int,
) -> float:
    """Stationary-initialized worst-case finite-horizon boundary probability."""
    pi = ge_stationary_distribution(T)
    vals = np.array([
        active_suppression_value(T, p_g, p_b, r, mu, horizon=horizon, gap=0, state=z, budget=budget)
        for z in (0, 1)
    ])
    return float(pi @ vals)


def robust_min_window(
    T: np.ndarray,
    p_g: float,
    p_b: float,
    r: float,
    *,
    horizon: int,
    budget: int,
    epsilon: float,
    mu_max: int = 50,
) -> int:
    """Smallest window meeting a finite-horizon adversarial failure target."""
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie in (0,1)")
    if horizon < 1 or budget < 0:
        raise ValueError("invalid horizon/budget")
    # If mu <= min(budget, horizon), the attacker can deterministically
    # suppress mu consecutive requests and reach the boundary.  Hence the
    # first nontrivial candidate for any epsilon < 1 is one step above that
    # deterministic region.
    lo = 1 + min(budget, horizon)
    if lo > mu_max:
        raise ValueError("no feasible window within mu_max")

    def risk(mu: int) -> float:
        return active_suppression_initial_value(
            T, p_g, p_b, r, mu, horizon=horizon, budget=budget
        )

    if risk(mu_max) > epsilon:
        raise ValueError("no feasible window within mu_max")

    hi = mu_max
    while lo < hi:
        mid = (lo + hi) // 2
        if risk(mid) <= epsilon:
            hi = mid
        else:
            lo = mid + 1
    return lo


def request_pmf(p: float, r: float, mu: int, *, t_max: int = 100) -> np.ndarray:
    """Exact total-request PMF coefficients up to t_max from the renewal PGF.

    Index 0 is included and equals zero because at least one request is needed.
    For practical parameter ranges, a sufficiently large t_max captures the entire
    mass to machine precision.
    """
    _check_prob(p, "p", strict_one=True)
    _check_prob(r, "r", strict_one=True)
    _check_mu(mu)
    if t_max < 1:
        raise ValueError("t_max must be positive")
    q = 1.0 - p
    a = np.zeros(mu + 1, dtype=float)
    for k in range(1, mu + 1):
        a[k] = q * p ** (k - 1)
    n = (1.0 - r) * a.copy()
    n[mu] += p ** mu
    g = np.zeros(t_max + 1, dtype=float)
    for t in range(1, t_max + 1):
        val = n[t] if t <= mu else 0.0
        for k in range(1, min(mu, t) + 1):
            val += r * a[k] * g[t - k]
        g[t] = val
    return g


def robust_system_frontier(
    T_c: np.ndarray, p_cg: float, p_cb: float, r_c: float, horizon_c: int, budget_c: int,
    T_s: np.ndarray, p_sg: float, p_sb: float, r_s: float, horizon_s: int, budget_s: int,
    *, epsilon_sys: float, mu_c_max: int = 30, mu_s_max: int = 30,
) -> list[tuple[int, int, float]]:
    """Pareto-minimal robust system windows under independent link risk models.

    Each link failure is the stationary-initialized finite-horizon worst-case value.
    Monotonicity in window size allows a binary search for the minimum sensor window
    compatible with each cluster-head window candidate.
    """
    if not (0.0 < epsilon_sys < 1.0):
        raise ValueError("epsilon_sys must lie in (0,1)")

    c_cache: dict[int, float] = {}
    s_cache: dict[int, float] = {}

    def Fc(mu: int) -> float:
        if mu not in c_cache:
            c_cache[mu] = active_suppression_initial_value(
                T_c, p_cg, p_cb, r_c, mu, horizon=horizon_c, budget=budget_c
            )
        return c_cache[mu]

    def Fs(mu: int) -> float:
        if mu not in s_cache:
            s_cache[mu] = active_suppression_initial_value(
                T_s, p_sg, p_sb, r_s, mu, horizon=horizon_s, budget=budget_s
            )
        return s_cache[mu]

    raw: list[tuple[int, int, float]] = []
    for mu_c in range(1, mu_c_max + 1):
        fc = Fc(mu_c)
        eps_s = sensor_failure_budget(fc, epsilon_sys)
        if eps_s <= 0.0:
            continue
        # Minimum sensor window with Fs <= residual budget.
        lo, hi = 1, mu_s_max
        if Fs(hi) > eps_s:
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if Fs(mid) <= eps_s:
                hi = mid
            else:
                lo = mid + 1
        mu_s = lo
        fsys = system_failure_probability(fc, Fs(mu_s))
        if fsys <= epsilon_sys * (1.0 + 1e-12):
            raw.append((mu_c, mu_s, fsys))

    # Remove componentwise dominated staircase repeats.
    out: list[tuple[int, int, float]] = []
    best_s = math.inf
    for row in raw:
        if row[1] < best_s:
            out.append(row)
            best_s = row[1]
    return out
