# A Mathematical Reliability Model for Multi Link Synchronization in the SI-AKA Scheme

This repository provides the source code and numerical results used to reproduce the mathematical analysis presented in the paper **A Mathematical Reliability Model for Multi Link Synchronization in the SI-AKA Scheme**.

For simplicity, **SI-AKA** refers to the published *Secure Authentication Scheme Using Diffie Hellman Key Agreement for Smart IoT Irrigation Systems*. The repository does not modify the original SI-AKA authentication protocol. It only implements the mathematical models used to study its synchronization reliability.

## Scope of the repository

The provided code reproduces the following parts of the analysis.

1. The finite state verification of the synchronization relation between the Agriculture Professional and the Authentication Center.
2. The independent request and response loss model for the two synchronization window relations.
3. The mathematical selection of the synchronization windows \(\mu_2\) and \(\mu_0\).
4. The first passage and recovery computation analysis.
5. The Gilbert Elliott model used to represent burst message losses.
6. The exact matrix renewal model and the absorbing Markov validation.
7. The bounded active message suppression model.
8. The Monte Carlo validation and the numerical data used in the reported results.

The communication parameters used in the numerical section are analysis parameters selected to study the behavior of the model. They are not measurements from a specific agricultural deployment.

## Repository structure

```text
SI-AKA-Mathematical-Reliability-Model/

README.md
REPRODUCIBILITY.md
RESULTS.md
LICENSE
requirements.txt

src/
    siaka_models.py
    run_experiments.py
    generate_paper_figures.py

tests/
    test_models.py
    test_properties.py

results/
    numerical result files

figures/
    static/
    generated/
    source/

.github/
    workflows/
        reproducibility.yml
```

## Quick verification

From the repository root, run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.run_experiments
python -m src.generate_paper_figures
```

The test suite verifies the main mathematical expressions, the state model, the Gilbert Elliott reduction properties, the matrix renewal formulation, the active suppression model, and the Monte Carlo agreement.

## Main result reproduced by the code

For the asymmetric communication condition used in the paper,

\[
(p_C,r_C)=(0.08,0.04),
\qquad
(p_S,r_S)=(0.12,0.06),
\]

and the system requirement

\[
\varepsilon_{\mathrm{sys}}=10^{-4},
\]

the model selects

\[
(\mu_2,\mu_0)=(4,5),
\]

with an overall loss of recoverability probability of approximately

\[
6.91\times10^{-5}.
\]

The remaining numerical results are summarized in `RESULTS.md`.

## Reproducibility

The exact commands, model parameters, simulation settings, and generated files are described in `REPRODUCIBILITY.md`.

## License

The source code is released under the MIT License.

## Citation

The final `CITATION.cff` file will be added after the author order and publication metadata are fixed. This avoids publishing incomplete or incorrect citation information.
