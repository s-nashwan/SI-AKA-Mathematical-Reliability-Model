from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'/'generated'
OUT.mkdir(exist_ok=True)
SRC=ROOT
sys.path.insert(0,str(ROOT))
from src.siaka_models import (
    iid_failure_probability, system_failure_probability,
    ge_transition_from_eta_kappa, robust_min_window,
    active_suppression_initial_value,
)

# ------------------------------------------------------------------
# Figure 4: actual primary feasible region, not illustrative data.
# ------------------------------------------------------------------
pC,rC,pS,rS,eps = 0.08,0.04,0.12,0.06,1e-4
mus=np.arange(1,11)
Z=np.zeros((10,10))
for i,mu0 in enumerate(mus):
    for j,mu2 in enumerate(mus):
        Fc=iid_failure_probability(pC,rC,int(mu2))
        Fs=iid_failure_probability(pS,rS,int(mu0))
        Z[i,j]=system_failure_probability(Fc,Fs)

fig,ax=plt.subplots(figsize=(8.2,5.7))
# two-region background using a masked contourf
X,Y=np.meshgrid(mus,mus)
region=(Z<=eps).astype(int)
ax.contourf(X,Y,region,levels=[-0.5,0.5,1.5],alpha=0.18)
# boundary on integer lattice: mark feasible and infeasible grid points subtly
feas=Z<=eps
ax.scatter(X[~feas],Y[~feas],s=20,marker='x',alpha=0.35,label='Infeasible pairs')
ax.scatter(X[feas],Y[feas],s=20,marker='o',facecolors='none',alpha=0.45,label='Feasible pairs')
# Unique Pareto-minimal pair for the primary asymmetric design
ax.scatter([4],[5],s=95,zorder=5,label=r'Pareto-minimal pair $(4,5)$')
ax.annotate(r'$(4,5)$',xy=(4,5),xytext=(4.35,5.35),arrowprops=dict(arrowstyle='->',lw=0.8))
ax.axvline(4,linestyle='--',linewidth=0.8,alpha=0.45)
ax.axhline(5,linestyle='--',linewidth=0.8,alpha=0.45)
ax.text(7.0,7.6,'Feasible region',ha='center',va='center',fontsize=11)
ax.text(2.4,2.5,'Infeasible region',ha='center',va='center',fontsize=11)
ax.set_xlim(0.5,10.5); ax.set_ylim(0.5,10.5)
ax.set_xticks(mus); ax.set_yticks(mus)
ax.set_xlabel(r'$\mu_2$  ($AuC$ to $CH_j$ window)')
ax.set_ylabel(r'$\mu_0$  ($CH_j$ to $AS_k$ window)')
ax.grid(True,linestyle='--',linewidth=0.5,alpha=0.35)
ax.legend(loc='upper left',fontsize=9,frameon=True)
fig.tight_layout()
fig.savefig(OUT/'fig4_actual.png',dpi=300,bbox_inches='tight')
fig.savefig(OUT/'fig4_actual.pdf',bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# Figure 6: analytical vs Monte Carlo parity, tested cases.
# ------------------------------------------------------------------
val=pd.read_csv(ROOT/'results/validation_cases.csv')
fig,ax=plt.subplots(figsize=(7.4,5.5))
for model,marker in [('IID','o'),('GE','s')]:
    d=val[val.model==model]
    yerr=np.vstack([d.monte_carlo-d.ci_low,d.ci_high-d.monte_carlo])
    ax.errorbar(d.analytical,d.monte_carlo,yerr=yerr,fmt=marker,linestyle='none',capsize=3,label=model,alpha=0.9)
lo=min(val.analytical.min(),val.ci_low.min())*0.75
hi=max(val.analytical.max(),val.ci_high.max())*1.25
ax.plot([lo,hi],[lo,hi],linestyle='--',linewidth=1,label='Exact agreement')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
ax.set_xlabel('Analytical loss of recoverability probability')
ax.set_ylabel('Monte Carlo estimate')
ax.grid(True,which='both',linestyle='--',linewidth=0.5,alpha=0.35)
ax.legend(loc='upper left',fontsize=9)
fig.tight_layout()
fig.savefig(OUT/'fig6_validation.png',dpi=300,bbox_inches='tight')
fig.savefig(OUT/'fig6_validation.pdf',bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# Figure 7: required window vs burst persistence at fixed mean loss .10.
# ------------------------------------------------------------------
pers=pd.read_csv(ROOT/'results/ge_persistence_design.csv')
fig,ax=plt.subplots(figsize=(7.5,5.2))
ax.step(pers.kappa,pers.required_window,where='post',linewidth=1.6)
ax.scatter(pers.kappa,pers.required_window,s=28)
# highlight representative conditions used in table
for k in [0.0,0.55,0.75,0.90]:
    row=pers.iloc[(pers.kappa-k).abs().argmin()]
    ax.annotate(f'{int(row.required_window)}',xy=(row.kappa,row.required_window),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
ax.set_xlabel(r'Burst persistence parameter $\kappa$')
ax.set_ylabel('Required synchronization window')
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(-0.02,0.97); ax.set_ylim(4.5,9.6)
ax.grid(True,linestyle='--',linewidth=0.5,alpha=0.35)
fig.tight_layout()
fig.savefig(OUT/'fig7_burst_window.png',dpi=300,bbox_inches='tight')
fig.savefig(OUT/'fig7_burst_window.pdf',bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------
# Figure 8: robust window vs active suppression budget, eps=1e-4.
# High-burst case kappa=.90, eta=.20, pG=.02, pB=.42, r=.05, H=20.
# ------------------------------------------------------------------
T=ge_transition_from_eta_kappa(.20,.90)
pG,pB,r=.02,.42,.05
H=20
budgets=np.arange(0,6)
windows=[]; probs=[]
for B in budgets:
    m=robust_min_window(T,pG,pB,r,horizon=H,budget=int(B),epsilon=1e-4,mu_max=50)
    v=active_suppression_initial_value(T,pG,pB,r,m,horizon=H,budget=int(B))
    windows.append(m); probs.append(v)
# save values to CSV for manuscript traceability
pd.DataFrame({'budget':budgets,'robust_window':windows,'failure_probability':probs}).to_csv(OUT/'section6_suppression_values.csv',index=False)
fig,ax=plt.subplots(figsize=(7.4,5.2))
ax.plot(budgets,windows,marker='o',linewidth=1.6)
for x,y in zip(budgets,windows):
    ax.annotate(str(y),xy=(x,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
ax.set_xlabel('Active message suppression budget $B$')
ax.set_ylabel('Required robust synchronization window')
ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(-0.2,5.2); ax.set_ylim(min(windows)-0.8,max(windows)+0.8)
ax.grid(True,linestyle='--',linewidth=0.5,alpha=0.35)
fig.tight_layout()
fig.savefig(OUT/'fig8_suppression_window.png',dpi=300,bbox_inches='tight')
fig.savefig(OUT/'fig8_suppression_window.pdf',bbox_inches='tight')
plt.close(fig)

print('Generated:', ', '.join(p.name for p in sorted(OUT.glob('fig[4678]_*.png'))))
print('Suppression:', list(zip(budgets,windows,probs)))
