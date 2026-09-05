# Reproducibility

This file explains how to reproduce the mathematical and numerical results reported in the SI-AKA synchronization study.

## 1. Verified environment

The current package was verified using:

```text
Python 3.13.5
NumPy 2.3.5
pandas 2.2.3
matplotlib 3.10.8
pytest 9.0.2
```

Install the required packages with:

```bash
python -m pip install -r requirements.txt
```

## 2. Verify the mathematical implementation

Run:

```bash
python -m pytest -q
```

The current package contains 34 automated tests. The tests verify the main closed form expressions, minimum synchronization window, request distribution, first passage quantities, AP to AuC synchronization bound, Gilbert Elliott properties, exact matrix renewal model, active suppression model, robust window selection, and Monte Carlo agreement.

## 3. Regenerate the numerical result files

Run:

```bash
python -m src.run_experiments
```

The command regenerates the CSV files in `results/`.

The main output must report the following values:

```text
Primary pair: (mu2,mu0)=(4,5)
Fsys = 6.91369117198e-05
All validation analytical values inside Wilson 95% intervals: True
GE mean loss: 0.100000
```

The exact matrix renewal model is also compared with the full absorbing Markov model over 250 deterministic stress cases. The expected differences are at numerical machine precision.

## 4. Regenerate the numerical figures used in the paper

Run:

```bash
python -m src.generate_paper_figures
```

This command regenerates the following numerical figures in `figures/generated/`:

1. `fig4_actual.pdf` and `fig4_actual.png`
2. `fig6_validation.pdf` and `fig6_validation.png`
3. `fig7_burst_window.pdf` and `fig7_burst_window.png`
4. `fig8_suppression_window.pdf` and `fig8_suppression_window.png`

Figures 1, 2, and 3 are explanatory synchronization diagrams and are stored in `figures/static/`. Figure 5 is generated from the TikZ source stored in `figures/source/`.

## 5. Basic asymmetric synchronization case

The independent loss analysis uses:

\[
p_C=0.08,
\qquad
r_C=0.04,
\]

for the Authentication Center to Cluster Head relation, and

\[
p_S=0.12,
\qquad
r_S=0.06,
\]

for the Cluster Head to Agriculture Sensor relation.

The required system loss of recoverability probability is

\[
\varepsilon_{\mathrm{sys}}=10^{-4}.
\]

The selected pair is

\[
(\mu_2,\mu_0)=(4,5).
\]

## 6. Burst message loss analysis

The Gilbert Elliott analysis uses a stationary bad state probability of

\[
\eta=0.20,
\]

with

\[
p_G=0.02,
\qquad
p_B=0.42.
\]

Therefore, the average request loss probability remains

\[
\bar p=0.10.
\]

The burst persistence parameter \(\kappa\) is varied while the stationary average loss remains unchanged. Representative results are:

```text
kappa = 0.00   required window = 5
kappa = 0.55   required window = 7
kappa = 0.75   required window = 8
kappa = 0.90   required window = 9
```

The response loss probability is \(r=0.05\), and the reliability requirement is \(10^{-4}\).

## 7. Active message suppression analysis

The active suppression experiment uses the high burst condition

\[
\eta=0.20,
\qquad
\kappa=0.90,
\qquad
p_G=0.02,
\qquad
p_B=0.42,
\qquad
r=0.05.
\]

The authentication horizon is

\[
H=20,
\]

and the required failure probability is

\[
10^{-4}.
\]

For suppression budgets from zero to five, the robust windows are:

```text
B = 0   mu = 9
B = 1   mu = 10
B = 2   mu = 11
B = 3   mu = 12
B = 4   mu = 13
B = 5   mu = 14
```

The corresponding values are stored in `results/section6_suppression_values.csv`.

## 8. Monte Carlo validation

The simulation uses fixed random seeds so that the reported results can be reproduced. The analytical probability is compared with the simulated probability using two sided 95% Wilson confidence intervals.

The complete validation cases are stored in:

```text
results/validation_cases.csv
```

All analytical values in the verified package fall inside the corresponding 95% confidence intervals.

## 9. Interpretation of the generated files

The repository separates mathematical results from explanatory figures. The CSV files are the numerical source of the reported result figures. Therefore, a reader can inspect the exact values without extracting them from the PDF plots.
