from pathlib import Path
import math
import numpy as np
import pandas as pd

from src.siaka_models import (
    iid_failure_probability, iid_min_window, expected_response_count,
    expected_extra_hashes, total_request_moments, conditional_first_passage_means,
    request_pmf, pareto_frontier, system_failure_probability,
    ge_transition_from_eta_kappa, ge_stationary_distribution, ge_mean_loss,
    ge_loss_autocorrelation, ge_spectral_radius, build_ge_absorbing_chain,
    ge_absorption_probabilities, ge_expected_requests, ge_expected_extra_hashes,
    ge_cycle_kernel, ge_matrix_renewal_metrics,
    active_suppression_initial_value, robust_min_window,
    iid_monte_carlo, ge_monte_carlo, wilson_interval, ap_reachable_states,
)

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
RESULTS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------
# 1) AP--AuC exhaustive abstract-state verification
# ---------------------------------------------------------------------
states=sorted(ap_reachable_states(max_depth=20))
pd.DataFrame(states,columns=['state','hash_gap']).to_csv(RESULTS/'ap_reachable_states.csv',index=False)

# ---------------------------------------------------------------------
# 2) Primary asymmetric SI-AKA design case
# ---------------------------------------------------------------------
pC,rC,pS,rS,eps_sys=0.08,0.04,0.12,0.06,1e-4
front=pareto_frontier(pC,rC,pS,rS,eps_sys,mu_c_max=15,mu_s_max=15)
if not front:
    raise RuntimeError('primary design case has no feasible pair')
mu2,mu0,Fsys=front[0]
rows=[]
for label,p,r,mu in [('AuC--CH',pC,rC,mu2),('CH--AS',pS,rS,mu0)]:
    F=iid_failure_probability(p,r,mu)
    mean,var=total_request_moments(p,r,mu)
    resp=expected_response_count(p,r,mu)
    hplus=expected_extra_hashes(p,r,mu)
    suc,bnd,rej=conditional_first_passage_means(p,r,mu)
    rows.append(dict(link=label,p_request=p,r_response=r,window=mu,
        failure_probability=F,mean_requests=mean,var_requests=var,
        mean_responses=resp,expected_extra_hashes=hplus,
        worst_case_extra_hashes=mu-1,
        mean_requests_given_success=suc,
        mean_requests_given_boundary=bnd,
        mean_requests_to_observed_rejection_given_boundary=rej))
primary=pd.DataFrame(rows)
primary['system_failure_probability']=Fsys
# Engineering conversion available explicitly for M2/M3 from SI-AKA source.
primary['expected_link_bits']=np.nan
primary['expected_extra_link_bits_vs_one_exchange']=np.nan
primary.loc[primary.link=='AuC--CH','expected_link_bits']=672*(primary.loc[primary.link=='AuC--CH','mean_requests']+primary.loc[primary.link=='AuC--CH','mean_responses'])
primary.loc[primary.link=='AuC--CH','expected_extra_link_bits_vs_one_exchange']=primary.loc[primary.link=='AuC--CH','expected_link_bits']-1344
# Original-paper legacy hash timing, only for comparability.
legacy_Th=0.00032
primary['legacy_expected_extra_hash_time_s']=primary['expected_extra_hashes']*legacy_Th
primary['legacy_worst_case_extra_hash_time_s']=primary['worst_case_extra_hashes']*legacy_Th
primary.to_csv(RESULTS/'primary_asymmetric_design.csv',index=False)

# ---------------------------------------------------------------------
# 3) IID design surface
# ---------------------------------------------------------------------
r_fixed=.05
surf=[]
for p in np.linspace(.02,.30,57):
    for mu in range(1,11):
        surf.append((p,mu,r_fixed,iid_failure_probability(float(p),r_fixed,mu)))
pd.DataFrame(surf,columns=['p_request','window','r_response','failure_probability']).to_csv(RESULTS/'iid_design_surface.csv',index=False)

# ---------------------------------------------------------------------
# 4) System feasible region + Pareto example with a visible staircase
# ---------------------------------------------------------------------
pCf,rCf,pSf,rSf,epsf=.16,.13,.24,.05,1e-3
feas=[]
for a in range(1,11):
    for b in range(1,11):
        Fc=iid_failure_probability(pCf,rCf,a)
        Fs=iid_failure_probability(pSf,rSf,b)
        feas.append((a,b,system_failure_probability(Fc,Fs)))
pd.DataFrame(feas,columns=['mu2','mu0','system_failure']).to_csv(RESULTS/'system_feasible_region.csv',index=False)
fr=pareto_frontier(pCf,rCf,pSf,rSf,epsf,mu_c_max=10,mu_s_max=10)
pd.DataFrame(fr,columns=['mu2','mu0','system_failure']).to_csv(RESULTS/'system_pareto_frontier.csv',index=False)

# ---------------------------------------------------------------------
# 5) PGF exact distribution and Monte Carlo validation
# ---------------------------------------------------------------------
p,r,mu=.18,.12,4
pmf=request_pmf(p,r,mu,t_max=50)
sim=iid_monte_carlo(p,r,mu,trials=600_000,seed=20260905)
# Independent MC histogram generated explicitly for the request count distribution.
rng=np.random.default_rng(20260906)
# reuse cycle process while preserving samples
counts=np.zeros(600_000,dtype=np.int16)
active=np.ones(600_000,dtype=bool)
while active.any():
    idx=np.flatnonzero(active)
    K=rng.geometric(1-p,size=idx.size)
    counts[idx]+=np.minimum(K,mu)
    boundary=K>mu
    active[idx[boundary]]=False
    di=idx[~boundary]
    if di.size:
        lost_resp=rng.random(di.size)<r
        active[di[~lost_resp]]=False
hist=np.bincount(counts,minlength=len(pmf))[:len(pmf)]/len(counts)
pd.DataFrame({'requests':np.arange(len(pmf)),'exact_pmf':pmf,'mc_pmf':hist}).to_csv(RESULTS/'pgf_request_distribution.csv',index=False)

# ---------------------------------------------------------------------
# 6) GE persistence at constant mean request-loss probability = 0.10
# ---------------------------------------------------------------------
eta,pG,pB,r=.20,.02,.42,.05
assert abs(ge_mean_loss(eta,pG,pB)-.10)<1e-14
ge_rows=[]
for kappa in np.linspace(0,.95,20):
    T=ge_transition_from_eta_kappa(eta,float(kappa)); pi=ge_stationary_distribution(T)
    rho=ge_spectral_radius(T,pG,pB)
    req=None; f_at_req=None
    for m in range(1,25):
        metrics=ge_matrix_renewal_metrics(T,pG,pB,r,m,pi=pi)
        f=metrics['failure_probability']
        if f<=1e-4:
            req=m;f_at_req=f;break
    ge_rows.append((kappa,rho,ge_loss_autocorrelation(eta,float(kappa),pG,pB,1),req,f_at_req))
pd.DataFrame(ge_rows,columns=['kappa','spectral_radius','lag1_loss_correlation','required_window','failure_at_required_window']).to_csv(RESULTS/'ge_persistence_design.csv',index=False)

# Detailed GE failure curves for representative persistence levels.
ge_curve=[]
for kappa in (0,.3,.5,.7,.85,.93):
    T=ge_transition_from_eta_kappa(eta,kappa); pi=ge_stationary_distribution(T)
    for m in range(1,13):
        metrics=ge_matrix_renewal_metrics(T,pG,pB,r,m,pi=pi)
        f=metrics['failure_probability']
        ge_curve.append((kappa,m,f,ge_spectral_radius(T,pG,pB)))
pd.DataFrame(ge_curve,columns=['kappa','window','failure_probability','spectral_radius']).to_csv(RESULTS/'ge_failure_curves.csv',index=False)

# ---------------------------------------------------------------------
# 7) Exact GE matrix-renewal compression vs 2*mu-state absorbing chain
#    Deterministic stress audit over a broad parameter range.
# ---------------------------------------------------------------------
mr_rows=[]
rng_audit=np.random.default_rng(20260905)
for case_id in range(250):
    eta_a=float(rng_audit.uniform(.05,.45))
    kappa_a=float(rng_audit.uniform(0,.95))
    pG_a=float(rng_audit.uniform(.001,.08))
    pB_a=float(rng_audit.uniform(max(pG_a+.05,.08),.65))
    r_a=float(rng_audit.uniform(0,.20))
    m=int(rng_audit.integers(1,11))
    T=ge_transition_from_eta_kappa(eta_a,kappa_a); pi=ge_stationary_distribution(T)
    mr=ge_matrix_renewal_metrics(T,pG_a,pB_a,r_a,m,pi=pi)
    Q,bf,bs=build_ge_absorbing_chain(T,pG_a,pB_a,r_a,m)
    f_chain,s_chain=ge_absorption_probabilities(Q,bf,bs,pi)
    req_chain=ge_expected_requests(Q,pi)
    hash_chain=ge_expected_extra_hashes(Q,pi,pG_a,pB_a,m)
    K,bvec=ge_cycle_kernel(T,pG_a,pB_a,m)
    row_balance=float(np.max(np.abs(K@np.ones(2)+bvec-np.ones(2))))
    mr_rows.append((case_id,eta_a,kappa_a,pG_a,pB_a,r_a,m,
                    mr['failure_probability'],f_chain,
                    mr['success_probability'],s_chain,
                    mr['mean_requests'],req_chain,
                    mr['expected_extra_hashes'],hash_chain,row_balance))
mrdf=pd.DataFrame(mr_rows,columns=[
    'case_id','eta','kappa','p_g','p_b','r_response','window',
    'matrix_failure','chain_failure','matrix_success','chain_success',
    'matrix_mean_requests','chain_mean_requests','matrix_extra_hashes','chain_extra_hashes',
    'cycle_probability_balance_error'])
mrdf['failure_abs_diff']=(mrdf.matrix_failure-mrdf.chain_failure).abs()
mrdf['success_abs_diff']=(mrdf.matrix_success-mrdf.chain_success).abs()
mrdf['requests_abs_diff']=(mrdf.matrix_mean_requests-mrdf.chain_mean_requests).abs()
mrdf['hash_abs_diff']=(mrdf.matrix_extra_hashes-mrdf.chain_extra_hashes).abs()
mrdf.to_csv(RESULTS/'ge_matrix_renewal_validation.csv',index=False)

# ---------------------------------------------------------------------
# 8) Bounded active suppression
# ---------------------------------------------------------------------
T=ge_transition_from_eta_kappa(.20,.90)
pG,pB,r=.02,.42,.05
H=20
eps_supp=1e-4
supp=[]
for B in range(0,6):
    for m in range(1,21):
        val=active_suppression_initial_value(T,pG,pB,r,m,horizon=H,budget=B)
        supp.append((B,m,H,val))
pd.DataFrame(supp,columns=['budget','window','horizon','worst_case_failure']).to_csv(RESULTS/'active_suppression_grid.csv',index=False)
rob=[]
for B in range(0,6):
    m=robust_min_window(T,pG,pB,r,horizon=H,budget=B,epsilon=eps_supp,mu_max=30)
    val=active_suppression_initial_value(T,pG,pB,r,m,horizon=H,budget=B)
    rob.append((B,m,val))
pd.DataFrame(rob,columns=['budget','robust_window','failure_probability']).to_csv(RESULTS/'active_suppression_robust_windows.csv',index=False)

# ---------------------------------------------------------------------
# 9) Analytical/Markov vs Monte Carlo validation cases
# ---------------------------------------------------------------------
validation=[]
iid_cases=[
    (.10,.05,2),(.10,.05,3),(.16,.08,3),(.12,.06,4),(.20,.10,4),(.25,.12,5)
]
for idx,(p,r,m) in enumerate(iid_cases):
    ana=iid_failure_probability(p,r,m)
    mc=iid_monte_carlo(p,r,m,trials=800_000,seed=100+idx)
    lo,hi=wilson_interval(mc['failures'],mc['trials'])
    validation.append(('IID',f'p={p}, r={r}, mu={m}',ana,mc['failure_probability'],lo,hi,mc['trials']))

ge_cases=[
    (.20,.50,.02,.42,.05,5),(.20,.70,.02,.42,.05,6),(.20,.85,.02,.42,.05,7),(.20,.93,.02,.42,.05,8)
]
for idx,(eta,kappa,pG,pB,r,m) in enumerate(ge_cases):
    T=ge_transition_from_eta_kappa(eta,kappa);pi=ge_stationary_distribution(T)
    ana=ge_matrix_renewal_metrics(T,pG,pB,r,m,pi=pi)['failure_probability']
    mc=ge_monte_carlo(T,pG,pB,r,m,trials=800_000,seed=300+idx)
    lo,hi=wilson_interval(mc['failures'],mc['trials'])
    validation.append(('GE',f'kappa={kappa}, mu={m}',ana,mc['failure_probability'],lo,hi,mc['trials']))
valdf=pd.DataFrame(validation,columns=['model','case','analytical','monte_carlo','ci_low','ci_high','trials'])
valdf['inside_95_ci']=(valdf.analytical>=valdf.ci_low)&(valdf.analytical<=valdf.ci_high)
valdf.to_csv(RESULTS/'validation_cases.csv',index=False)

# ---------------------------------------------------------------------
# 10) Summary text
# ---------------------------------------------------------------------
summary=[]
summary.append(f'Primary pair: (mu2,mu0)=({mu2},{mu0}), Fsys={Fsys:.12g}')
summary.append(f'AP reachable states: {states}')
summary.append(f'All validation analytical values inside Wilson 95% intervals: {bool(valdf.inside_95_ci.all())}')
summary.append(f'GE mean loss: {ge_mean_loss(eta,pG,pB):.6f}')
summary.append(f'GE matrix-renewal max failure diff vs absorbing chain: {mrdf.failure_abs_diff.max():.3e}')
summary.append(f'GE matrix-renewal max request-count diff vs absorbing chain: {mrdf.requests_abs_diff.max():.3e}')
(RESULTS/'SUMMARY.txt').write_text('\n'.join(summary)+'\n')
print('\n'.join(summary))
