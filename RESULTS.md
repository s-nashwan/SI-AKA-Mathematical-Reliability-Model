# Main Numerical Results

This file summarizes the numerical values reproduced by the repository. The values correspond to the analysis reported in the paper and are generated directly from the mathematical model.

## 1. AP to AuC synchronization bound

The exhaustive state analysis reaches only the following abstract states:

```text
S   synchronization gap = 0
P   synchronization gap = 0
A   synchronization gap = 1
```

Therefore, the reachable synchronization difference satisfies

\[
D_A\leq1.
\]

The corresponding result is stored in `results/ap_reachable_states.csv`.

## 2. Joint selection of \(\mu_2\) and \(\mu_0\)

The primary asymmetric communication condition is:

\[
(p_C,r_C)=(0.08,0.04),
\qquad
(p_S,r_S)=(0.12,0.06).
\]

For

\[
\varepsilon_{\mathrm{sys}}=10^{-4},
\]

the selected windows are

\[
\mu_2=4,
\qquad
\mu_0=5.
\]

The link probabilities are:

\[
F_C(4)=4.2666593849\times10^{-5},
\]

\[
F_S(5)=2.6471447317\times10^{-5}.
\]

The resulting system probability is

\[
F_{\mathrm{sys}}(4,5)=6.9136911720\times10^{-5}.
\]

The expected additional hash computations are approximately

\[
E[H_C^+]=0.0904,
\qquad
E[H_S^+]=0.1449.
\]

The maximum additional hash computations are three and four, respectively.

The complete values are stored in `results/primary_asymmetric_design.csv`.

## 3. Burst message losses

The average request loss probability is fixed at

\[
\bar p=0.10.
\]

The required synchronization window changes with burst persistence:

| Burst persistence \(\kappa\) | Required window |
| ---: | ---: |
| 0.00 | 5 |
| 0.55 | 7 |
| 0.75 | 8 |
| 0.90 | 9 |

Therefore, the same average loss probability can require a considerably larger synchronization window when the losses become more persistent.

The complete design curve is stored in `results/ge_persistence_design.csv`.

## 4. Exact matrix renewal validation

The two state matrix renewal formulation was compared with the full absorbing Markov model over 250 deterministic parameter configurations.

The maximum absolute difference in the failure probability is approximately

\[
5.55\times10^{-17}.
\]

The maximum absolute difference in the expected request count is approximately

\[
1.11\times10^{-15}.
\]

These values are at numerical machine precision and confirm the equivalence of the two implementations for the evaluated cases.

The complete comparison is stored in `results/ge_matrix_renewal_validation.csv`.

## 5. Active message suppression

For the high burst communication condition and an authentication horizon of \(H=20\), the robust synchronization windows are:

| Suppression budget \(B\) | Required robust window | Worst case probability |
| ---: | ---: | ---: |
| 0 | 9 | \(4.42\times10^{-5}\) |
| 1 | 10 | \(4.81\times10^{-5}\) |
| 2 | 11 | \(5.20\times10^{-5}\) |
| 3 | 12 | \(5.59\times10^{-5}\) |
| 4 | 13 | \(5.98\times10^{-5}\) |
| 5 | 14 | \(6.36\times10^{-5}\) |

These values are stored in `results/section6_suppression_values.csv`.

## 6. Monte Carlo validation

The analytical and exact state results were compared with Monte Carlo simulation under both independent and Gilbert Elliott loss conditions.

All analytical values used in the validation are inside their corresponding two sided 95% Wilson confidence intervals.

The complete comparison is stored in `results/validation_cases.csv`.
