# Label shift and anchor-based trust in federated logistic regression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](requirements.txt)
[![Data: ACS PUMS 2018](https://img.shields.io/badge/Data-US%20Census%20PUMS-green.svg)](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/)
[![Artifact: Reproducible](https://img.shields.io/badge/Artifact-Deterministic%20(20%20Seeds)-orange.svg)](#reproducibility--verification)

> Empirical software artifact and evaluation suite for:  
> **"Label shift and anchor-based trust in federated logistic regression: exclusion, estimand bias and a centered correction"**  
> *Toqeer Ali Syed, Muhammad Shoaib Siddiqui, and Ali Akarma*  
> Submitted to the *Journal of Applied Statistics*.

---

## Table of Contents

- [Overview & Problem Formulation](#overview--problem-formulation)
  - [The Mechanism: Cosine Collapse Under Label Shift (Proposition 1)](#the-mechanism-cosine-collapse-under-label-shift-proposition-1)
  - [Algorithmic Corrections: Centered Cosine & CPQ-B](#algorithmic-corrections-centered-cosine--cpq-b)
- [Repository Architecture](#repository-architecture)
- [Experimental Matrix](#experimental-matrix)
  - [Implemented Aggregation Rules](#implemented-aggregation-rules)
  - [Adversarial Threat Models](#adversarial-threat-models)
  - [Benchmark Datasets](#benchmark-datasets)
- [Environment Setup](#environment-setup)
  - [Dependencies](#dependencies)
  - [Virtual Environment Setup](#virtual-environment-setup)
- [Quick Start: Instant Verification (< 10 Seconds)](#quick-start-instant-verification--10-seconds)
  - [1. Regression & Reproduction Sanity Check](#1-regression--reproduction-sanity-check)
  - [2. Synthetic Prototype Simulation](#2-synthetic-prototype-simulation)
  - [3. Proposition 1 Mechanism Check](#3-proposition-1-mechanism-check)
- [Full Empirical Reproduction](#full-empirical-reproduction)
  - [1. Data Extraction (Optional Rebuild)](#1-data-extraction-optional-rebuild)
  - [2. Experiment Execution Suite](#2-experiment-execution-suite)
  - [3. Statistical Analysis & Table Generation](#3-statistical-analysis--table-generation)
  - [4. Manuscript Figure Data Generation](#4-manuscript-figure-data-generation)
- [Reproducibility & Verification Guarantee](#reproducibility--verification-guarantee)
- [Benchmark Data & Ethical Fairness Audit](#benchmark-data--ethical-fairness-audit)
- [Citation](#citation)
- [License](#license)

---

## Overview & Problem Formulation

Federated aggregation schemes that defend against Byzantine and untrusted updates often rely on a clean, centralized **anchor dataset** (e.g., FLTrust). In these frameworks, the central server computes the cosine similarity between each local client's gradient update $g_k$ and the anchor update $g_0$, weighting clients proportionally to $\max(0, \cos(g_k, g_0))$.

This repository provides the complete, deterministic empirical codebase demonstrating that **under label shift, anchor-based trust filters penalize and exclude benign institutions solely based on their local outcome base rate.**

### The Mechanism: Cosine Collapse Under Label Shift (Proposition 1)

For federated logistic regression with cross-entropy loss, client gradients take the form $g_k = \frac{1}{n_k} \sum_{i} x_i (p_i - y_i)$. When an institution differs from the anchor primarily in its outcome base rate ($\Delta_k = \bar{y}_k - \bar{y}_0$), its residuals shift systematically by approximately $-\Delta_k$, causing its local gradient to diverge along the mean feature direction $m = \mathbb{E}[x]$:

$$g_k \approx g_0 - \Delta_k m$$

The cosine similarity between the client update and the anchor update is analytically governed by:

$$\cos(g_k, g_0) = \frac{1 - b d_k}{\sqrt{1 - 2 b d_k + d_k^2}}, \quad \text{where} \quad b = \frac{g_0^\top m}{\|g_0\| \|m\|}, \quad d_k = \frac{\Delta_k \|m\|}{\|g_0\|}$$

When $b \approx 0$, this cosine decays monotonically with the magnitude of the base-rate discrepancy $|d_k|$. As a consequence:
1. **Benign Institutions Are Downweighted or Excluded**: Clients in low-employment or high-employment regions are penalized simply because their local demographic base rates deviate from the anchor's base rate.
2. **Estimand Bias & Calibration Distortion**: Excluding these institutions skews the pooled model's predictions, inflating Calibration-in-the-Large (CITL) error and hurting downstream subgroup fairness.

### Algorithmic Corrections: Centered Cosine & CPQ-B

The codebase investigates concrete, privacy-preserving remedies:
- **Centered Cosine (`center`)**: Removes the base-rate gradient projection by centering client gradients ($g_k - (g_k^\top e_b) m_k$), eliminating the base-rate gap penalty without requiring private demographic disclosure.
- **Coverage-Preserving Quarantine (CPQ-B / `cpqj`)**: An audit-driven quarantine mechanism that replaces raw gradient alignment with a balanced paired performance test ($J = \text{TPR} - \text{FPR}$ across protected groups), paired with Bonferroni-adjusted readmission bounds.
- **Group-Wise Release Gate**: A deterministic gateway evaluating whether candidate global models degrade worst-case subgroup performance prior to round deployment.

---

## Repository Architecture

```
.
├── code/
│   ├── experiment.py            # Synthetic prototype generator, learner, and prototype baseline
│   ├── prepare_acs.py           # Census CSV parser -> data/acs/*.npz (features, weights, ancestry)
│   ├── revision_experiments.py  # Full 20-seed experiment execution suite (all parts & ablations)
│   ├── revision_analysis.py     # Paired bootstrap intervals, sensitivity analysis, LaTeX tables
│   ├── jas_analysis.py          # Clustered PUMA regressions, permutation tests, calibration, tables
│   ├── jas_mechanism.py         # Numerical verification of Proposition 1 (mechanism check)
│   ├── jas_figures.py           # Generates TikZ/PGFPlots data files for manuscript figures
│   ├── check_reproduce.py       # Instant verification script (tests 8 core runs to 1e-12 tolerance)
│   └── flatten_submission.py    # Self-contained LaTeX package flattener
├── data/
│   └── acs/
│       ├── raw/                 # ACS 2018 1-year PUMS zip archives (13 states) + SHA256SUMS
│       ├── acs_<state>.npz      # Preprocessed features X, label y, audit label g, PUMA IDs
│       ├── acs_<state>_w.npz    # Person weights (PWGTP) and 80 replicate weights (PWGTP1-80)
│       └── acs_<state>_anc.npz  # Features augmented with one-hot ancestry categories (ANC)
├── results/
│   ├── runs.csv                 # Prototype benchmark run log
│   ├── summary.json             # Prototype summary metrics
│   ├── policy_checks.json       # Deterministic policy validation results
│   ├── revision/                # Benchmark runs, ablation logs, and LaTeX tables for Revision 1
│   └── revision_jas/            # Full JAS empirical outputs (20 seeds), timing records, tables,
│                                # and pre-registration protocol (heldout_prereg.md)
├── CITATION.cff                 # Machine-readable academic citation metadata
├── LICENSE                      # MIT Open-Source License
├── requirements.txt             # Primary environment dependencies
└── README.md                    # Repository documentation
```

> [!NOTE]
> The LaTeX manuscript sources (`Paper/`) are maintained in an internal workspace and are deliberately excluded via `.gitignore`. All numerical findings and tables in the paper are reproducible directly from the `code/` and `results/` trees in this repository.

---

## Experimental Matrix

### Implemented Aggregation Rules

| Rule Identifier | Method Name | Description / Strategy | Category |
| :--- | :--- | :--- | :--- |
| `fedavg` | **FedAvg** | Standard sample-weighted parameter averaging (McMahan et al., 2017) | Unweighted Baseline |
| `median` | **Coordinate Median** | Dimension-wise median of client parameter updates (Yin et al., 2018) | Coordinate-wise Robust |
| `trimmed` | **Trimmed Mean** | Coordinate-wise trimmed average removing extreme client updates | Coordinate-wise Robust |
| `multikrum` | **Multi-Krum** | Distance-based filtering selecting updates closest to neighbors (Blanchard et al., 2017) | Distance-based Robust |
| `fltrust` | **FLTrust** | Cosine similarity against anchor update with norm clipping (Cao et al., 2021) | Anchor Trust (Standard) |
| `anchor` | **Anchor Rule** | Prototype trust filter using unscaled cosine weighting against anchor | Anchor Trust (Prototype) |
| `fedval` | **FedVal** | Client-side validation score weighting on local validation splits (Valadi et al., 2023) | Client Validation |
| `cpq` | **CPQ-R** | Coverage-Preserving Quarantine using recall-based paired significance testing | Quarantine / Audited |
| `cpqj` | **CPQ-B** | Coverage-Preserving Quarantine using Youden's $J = \text{TPR} - \text{FPR}$ balanced test | Quarantine / Audited |
| `singh` | **Microaggregation** | Microaggregation-based Byzantine anomaly detector (Singh et al., 2023) | Benchmark Baseline |

### Adversarial Threat Models

The evaluation tests client federations under Byzantine compromise across 20 fixed random seeds (seeds `1001`–`1020`), rotating attacker identities across every client position:
1. **Sign Flip (`signflip`)**: Malicious client reverses update vector direction ($\Delta w \leftarrow -c \cdot \Delta w$).
2. **Targeted Subgroup Suppression (`audit_suppress`)**: Attacker selectively perturbs gradients to suppress true-positive utility on the protected disability subgroup.
3. **Min-Max Attack (`minmax`)**: Targeted perturbation bounding deviation within maximum distance between honest updates (Shejwalkar & Houmansadr, 2021).
4. **Positive Push (`push`)**: Directional bias shift designed to circumvent validation coverage filters.
5. **Lying Reporter (`liar`)**: Byzantine client submits honest weights but broadcasts falsified validation metrics.
6. **Slow-Start Sign Flip (`slow_signflip`)**: Adversary remains benign in early rounds and initiates aggressive poisoning mid-training.
7. **Two-Attacker Stress Test (`two_attackers`)**: Multiple simultaneously compromised clients in a 6-institution network.

### Benchmark Datasets

- **Synthetic Proxy Benchmark**: Controlled 10-dimensional feature space across 4 institutions with parameterized label shift and group allocations (`code/experiment.py`).
- **American Community Survey (ACS) State Federation**: Six development states selected for demographic variance:
  - **West Virginia (`wv`)**, **Maine (`me`)**, **New Mexico (`nm`)**, **Utah (`ut`)**, **North Dakota (`nd`)**, **Mississippi (`ms`)**.
  - **Anchor State**: **Vermont (`vt`)** ($n_0 \approx 400$ anchor records).
- **Held-Out Validation States**: Six strictly held-out states evaluated under a frozen pre-registered protocol (`heldout_prereg.md`):
  - **Wyoming (`wy`)**, **Alaska (`ak`)**, **South Dakota (`sd`)**, **Delaware (`de`)**, **Rhode Island (`ri`)**, **Montana (`mt`)**.
- **Small-Area (PUMA) Federation**: 89 distinct Public Use Microdata Areas (PUMAs) across the development states, evaluating high-granularity spatial label shift.

---

## Environment Setup

### Dependencies

- **Python**: `3.10.11` (compatible with Python $\ge 3.10$)
- **Core Libraries**: `numpy==2.2.6`, `pandas==2.3.3`
- **Statistical Analysis**: `scipy>=1.15.0` (required for bootstrap and hypothesis testing in `jas_analysis.py`)
- **Optional**: `matplotlib` (only needed if generating visual plots from `experiment.py`)

### Virtual Environment Setup

#### Linux / macOS (Bash / Zsh)

```bash
# Clone the repository
git clone https://github.com/aliakarma/TrustFair-Agent.git
cd TrustFair-Agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt scipy
```

#### Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/aliakarma/TrustFair-Agent.git
cd TrustFair-Agent

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install required dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt scipy
```

---

## Quick Start: Instant Verification (< 10 Seconds)

To verify repository integrity without running the complete multi-hour experiment suite, execute the deterministic validation targets below.

### 1. Regression & Reproduction Sanity Check

`code/check_reproduce.py` runs 8 representative cases across synthetic and ACS benchmarks (FedAvg, FLTrust, FedVal, CPQ, liar attack, sign flip) and compares output parameters against the committed baseline (`results/revision/runs.csv`) at absolute tolerance $\epsilon \le 10^{-12}$:

```bash
python code/check_reproduce.py
```

*Expected output:*
```text
acs 1001 none fedavg f1 0.7449 rel_f1 0.7449 gate 0
acs 1001 none cpqj f1 0.7408 rel_f1 0.7396 gate 3
acs 1002 signflip cpqj f1 0.7503 rel_f1 0.7503 gate 0
acs 1001 none fltrust f1 0.7293 rel_f1 0.7293 gate 0
acs 1003 liar cpqj f1 0.7391 rel_f1 0.6974 gate 30
acs 1001 none fedval f1 0.7413 rel_f1 0.7413 gate 0
synthetic 1001 signflip anchor f1 0.8128 rel_f1 0.8128 gate 0
synthetic 1004 none cpq f1 0.7953 rel_f1 0.7953 gate 0
mismatches: 0
```

### 2. Synthetic Prototype Simulation

Executes the prototype 4-client federated learning benchmark with adversarial injection and deterministic capability gating (~3 seconds):

```bash
python code/experiment.py
```

### 3. Proposition 1 Mechanism Check

Compares observed gradient cosine alignments with theoretical predictions derived from base-rate discrepancies ($\Delta_k$) across rounds 1, 5, 10, 20, and 30:

```bash
# Linux / macOS
TF_OUT=results/revision_jas python code/jas_mechanism.py

# Windows (PowerShell)
$env:TF_OUT="results/revision_jas"; python code/jas_mechanism.py
```

Outputs: `mechanism_check.json`, `mechanism_scatter.dat`, and `mechanism_points.dat`.

---

## Full Empirical Reproduction

All commands must be executed from the **repository root**.

> [!TIP]
> **Precomputed Arrays Already Included**: All raw census archives have been preprocessed into `data/acs/*.npz`. Step 1 (`prepare_acs.py`) is only needed if you wish to rebuild the derived feature arrays from scratch.

### 1. Data Extraction (Optional Rebuild)

Reconstructs the 13-state NumPy feature matrices, person weights, and ancestry splits from `data/acs/raw/*.zip`:

```bash
python code/prepare_acs.py
```

### 2. Experiment Execution Suite

The experiment suite spans 20 seeds across 13 modular partitions.

#### Linux / macOS (Bash)

```bash
# Core state federation benchmarks and baselines
TF_OUT=results/revision_jas python code/revision_experiments.py --part audit,main,g
TF_OUT=results/revision_jas python code/revision_experiments.py --part anchor,match
TF_OUT=results/revision_jas python code/revision_experiments.py --part puma
TF_OUT=results/revision_jas python code/revision_experiments.py --part mlp
TF_OUT=results/revision_jas python code/revision_experiments.py --part sens
TF_OUT=results/revision_jas python code/revision_experiments.py --part variants,null,sens2
TF_OUT=results/revision_jas python code/revision_experiments.py --part center,center_anchor,center_puma,heldout_center
TF_OUT=results/revision_jas python code/revision_experiments.py --part weighted,anc,singh
TF_OUT=results/revision_jas python code/revision_experiments.py --part heldout
TF_OUT=results/revision_jas python code/revision_experiments.py --part disclose
TF_OUT=results/revision_jas python code/revision_experiments.py --part partial
TF_OUT=results/revision_jas python code/revision_experiments.py --part partial_puma
```

#### Windows (PowerShell)

```powershell
$env:TF_OUT = "results/revision_jas"
python code/revision_experiments.py --part audit,main,g
python code/revision_experiments.py --part anchor,match
python code/revision_experiments.py --part puma
python code/revision_experiments.py --part mlp
python code/revision_experiments.py --part sens
python code/revision_experiments.py --part variants,null,sens2
python code/revision_experiments.py --part center,center_anchor,center_puma,heldout_center
python code/revision_experiments.py --part weighted,anc,singh
python code/revision_experiments.py --part heldout
python code/revision_experiments.py --part disclose
python code/revision_experiments.py --part partial
python code/revision_experiments.py --part partial_puma
```

> [!NOTE]
> **Computational Budget**: Running all experimental parts takes approximately **5.2 CPU-hours** on an AMD Ryzen 5 5600H processor. Parts are independent and can be executed concurrently in separate terminal sessions. Execution timing and platform metadata are automatically written to `results/revision_jas/env_<parts>.json`.

### 3. Statistical Analysis & Table Generation

Generates paired bootstrap confidence intervals (10,000 resamples), clustered regressions, calibration metrics, and publication-ready LaTeX tables in `results/revision_jas/tables/`:

```bash
# Linux / macOS
TF_RES=results/revision_jas python code/revision_analysis.py
TF_RES=results/revision_jas python code/jas_analysis.py

# Windows (PowerShell)
$env:TF_RES = "results/revision_jas"
python code/revision_analysis.py
python code/jas_analysis.py
```

### 4. Manuscript Figure Data Generation

Computes institution-level mean weights under standard vs. centered cosine aggregation, exporting data files (`weights_center_acs.dat`, `puma_center.dat`) for TikZ/PGFPlots visualization:

```bash
# Linux / macOS
TF_RES=results/revision_jas python code/jas_figures.py

# Windows (PowerShell)
$env:TF_RES = "results/revision_jas"
python code/jas_figures.py
```

---

## Reproducibility & Verification Guarantee

- **Deterministic Execution**: All seed sequences (`1001` to `1020`) and sample splits are hardcoded.
- **Zero Hallucination or Extrapolation**: Every metric, confidence interval, and $p$-value reported in the manuscript corresponds to an exact entry in `results/revision_jas/jas_numbers.json` and `results/revision_jas/tables/*.tex`.
- **Integrity Checksums**: Raw microdata archives in `data/acs/raw/` are tracked and validated against `data/acs/raw/SHA256SUMS`.
- **Pre-Registration**: The hypotheses and analytical protocol for the 6 held-out validation states were frozen prior to model evaluation and are documented in [results/revision_jas/heldout_prereg.md](results/revision_jas/heldout_prereg.md).

---

## Benchmark Data & Ethical Fairness Audit

The benchmark utilizes the **American Community Survey (ACS) 2018 1-Year Public Use Microdata Sample (PUMS)** released by the United States Census Bureau:
- **Task Definition**: Binary classification of employment status (`ESR == 1`, employed) among civilian individuals aged 16–90, adopting the standardized ACSEmployment definition (Ding et al., 2021).
- **Subgroup Fairness Audit**: Disability indicators (`DIS` disability recode, `DEAR` hearing difficulty, `DEYE` vision difficulty, `DREM` cognitive difficulty) are **strictly excluded from model features ($X$)** during training. They serve solely as post-hoc auditing labels to measure disparate treatment and error rates across demographic subgroups.
- **Ethical Considerations**: Historical employment patterns embody structural exclusion. These datasets are deployed strictly as empirical benchmarks for evaluating federated aggregation mechanisms under spatial and demographic shift, not as predictive tools for real-world employment decisions.

---

## Citation

If you use this codebase, empirical benchmarks, or algorithmic corrections in your research, please cite the associated manuscript:

```bibtex
@article{syed2026labelshift,
  title     = {Label shift and anchor-based trust in federated logistic regression: exclusion, estimand bias and a centered correction},
  author    = {Syed, Toqeer Ali and Siddiqui, Muhammad Shoaib and Akarma, Ali},
  journal   = {Journal of Applied Statistics},
  year      = {2026},
  note      = {Submitted. Software and data repository: \url{https://github.com/aliakarma/TrustFair-Agent}}
}
```

Detailed metadata is also available in [CITATION.cff](CITATION.cff).

---

## License

- **Source Code**: Released under the [MIT License](LICENSE).
- **Census Microdata**: ACS PUMS microdata files in `data/acs/raw/` are published by the US Census Bureau as public-domain United States Government works.
