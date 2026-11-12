# Anchor trust under label shift

Code, data and results for the paper *Anchor trust under label shift: whom robust federated aggregation excludes, with evidence from census employment data* (Toqeer Ali Syed, Muhammad Shoaib Siddiqui and Ali Akarma), submitted to the Journal of Applied Statistics. The title is a working title.

Robust aggregation rules that trust a small anchor dataset weight each institution by the cosine between its update and the anchor's. The paper shows that for federated logistic regression under label shift this cosine depends on an institution only through the gap between its outcome base rate and the anchor's. It measures the consequence on American Community Survey microdata (six development states, 89 small areas and six held-out states), shows that the exclusion biases the pooled model's calibration, and tests a centered cosine that removes the base-rate direction. It also evaluates coverage-preserving quarantine with a balanced paired test (CPQ-B) and a group-wise release gate.

Every number in the paper and its supplement is generated from the runs in this repository. Nothing is projected. The applicants in the supplement's use case are fictional, and no LLM, user study, secure aggregation or differential privacy was run.

## Layout

```
code/
  experiment.py            Prototype generator, learner and original FedAgent-Chain run
  prepare_acs.py           Census zip files -> data/acs/acs_<state>.npz (+ _w person weights, _anc with ancestry)
  revision_experiments.py  All experiments (20 seeds). Output folder from TF_OUT (default results/revision/)
  revision_analysis.py     Main summaries, paired bootstrap intervals, LaTeX tables (input folder from TF_RES)
  jas_analysis.py          Analyses added for the JAS version: permutation tests, clustered PUMA regression,
                           calibration, null federation, release gate, held-out hypotheses, jas_numbers.json
  jas_mechanism.py         Numerical check of Proposition 1 (mechanism_check.json, mechanism_scatter.dat)
  check_reproduce.py       Confirms that the current code reproduces the earlier runs exactly
  flatten_submission.py    Builds a flattened LaTeX upload package (written for the earlier Springer version)
data/acs/
  raw/                     ACS 2018 1-year PUMS person files for 13 states, with SHA256SUMS
  acs_<state>.npz          X (features), y (employed), g (disability, audit only), puma
  acs_<state>_w.npz        Person weight PWGTP and the 80 replicate weights
  acs_<state>_anc.npz      Features with ancestry (ANC) added, for the sensitivity run
results/
  runs.csv, summary.json, policy_checks.json   Original prototype run
  revision/                Runs of the earlier version (unchanged)
  revision_jas/            Every run, table and number of the JAS version, the held-out pre-specification
                           (heldout_prereg.md) and the per-part timing records (env_*.json)
Paper/
  main_jas.tex, supplementary_jas.tex, references.bib, figures/
```

## Setup

Python 3.10.11 was used. The exact package versions are in `requirements.txt`.

```
python -m pip install -r requirements.txt
```

## Reproduce

Run every command from the repository root, because the scripts resolve all paths from there.

```
python code/prepare_acs.py                                   # census zips -> data/acs/*.npz (13 states)
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
TF_OUT=results/revision_jas python code/jas_mechanism.py
TF_RES=results/revision_jas python code/revision_analysis.py
TF_RES=results/revision_jas python code/jas_analysis.py
python code/check_reproduce.py
```

The parts can run in parallel processes. Together they took about 5.2 CPU-hours on an AMD Ryzen 5 5600H. Each part writes its time to `results/revision_jas/env_<parts>.json`. Seeds 1001 to 1020 are fixed in the code. The derived arrays are already in `data/acs/`, so `prepare_acs.py` is only needed to rebuild them from the raw census files.

## What is implemented

- Nine aggregation rules: FedAvg, coordinate median, trimmed mean, Multi-Krum, FLTrust, FedVal, the prototype anchor rule, and CPQ with the recall test (CPQ-R) and with the balanced test (CPQ-B), plus the microaggregation detector of Singh et al. (2023).
- Trust-score variants: intercept left out of the cosine, centered cosine, anchor reweighted to the federation's base rate, pooled anchor, and an interval guard for CPQ-B.
- A group-wise release gate and simulated voluntary disclosure of the audit label.
- Attacks: sign flip, targeted suppression of the support group, min-max (two perturbation directions), positive push, a lying reporter, a slow-start sign flip and two attackers.
- Data: a synthetic proxy, a six-state ACS federation, an 89-PUMA federation, six held-out states, a null federation, matched resampling, person weights, an ancestry sensitivity run and partial participation (4 of 6 states or half of the PUMAs per round).

The worst-group training objective, the procedural-access audit metric, the accommodation state and the capability gateway for agents are specified in the supplement only. They are not in this code.

## Data

The data are the American Community Survey 2018 1-year Public Use Microdata Sample, published by the US Census Bureau at <https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/>. The files are public and contain no private records. `data/acs/raw/SHA256SUMS` lists the hashes of the downloaded zip files.

The task follows the ACSEmployment definition of Ding et al. (2021), restricted to ages 16 to 90. The label is employment status (`ESR == 1`). The disability indicators (`DIS`, `DEAR`, `DEYE`, `DREM`) are kept out of the model input and used only to audit results by group. The labels record employment status, which carries a history of exclusion, so the data serve as a benchmark for aggregation behavior and not as evidence about job matching.

## Citation

See `CITATION.cff`. The archive DOI will be added here once a release has been deposited.

## License

The code is released under the MIT License (see `LICENSE`). The census files in `data/acs/raw/` are US Government works and carry no copyright restriction.
