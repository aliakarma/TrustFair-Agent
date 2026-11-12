# Held-out replication: pre-registration

Written 2026-11-11, after the six ACS 2018 one-year person files were downloaded and before any of their records was read or any run was made on them.

## States

Wyoming, Alaska, South Dakota, Delaware, Rhode Island and Montana (`HELDOUT` in `code/revision_experiments.py`). Rule: six small states that were not used anywhere during method development (the development states were WV, ME, NM, UT, ND, MS and the anchor VT). Small states keep the run time of the full grid low. Their employment rates and disability prevalences were unknown to us when they were chosen.

## Design (identical to the main ACS grid)

Same code path (`grid('acs')`), same anchor (Vermont, 500 records), same 20 seeds (1001 to 1020), same splits per state (2,000 train, 1,000 test, 500 validation), 30 rounds, all nine rules, clean condition and rotated single attackers (sign-flip, targeted, min-max, positive push, lying reporter), all CPQ parameters as fixed in the paper (c = 2, gamma_0 = 0.2, eta_gamma = 10, eta_zeta = 5, tau = 0.5, z = 1.96, delta = 0.01).

## Hypotheses, stated before the run

- **H1 (mechanism).** In the clean condition, for each of FLTrust, the anchor rule and CPQ-R, a state's mean aggregation weight falls as the absolute gap between its employment rate and Vermont's grows. Test: Spearman rank correlation between the gap and the mean weight over the six states, one-sided (negative), exact permutation p-value, reported per rule. With six states and a small spread of rates, the test may have low power, and a null result will be reported as such.
- **H2 (CPQ-B coverage).** In the clean condition, CPQ-B's lowest state weight is higher than the anchor rule's lowest state weight (mean over seeds, paired by seed, 95% bootstrap interval of the difference).
- **H3 (CPQ-B attack resistance).** Under sign-flip, CPQ-B readmits the attacker in fewer rounds per run than CPQ-R (paired by seed).
- **H4 (cost of CPQ-B).** In the clean condition, CPQ-B's support-group Youden index J_S is not higher than FedAvg's (paired difference interval includes or lies below zero), as in the development states.

Every outcome will be reported, including outcomes that contradict H1 to H3.

## Addendum, 2026-11-11 (added while the held-out grid was running, before any held-out output was read)

The centered trust cosine (`FLAGS['center']`, the base-rate direction removed before the cosine, motivated by Proposition 1) was designed on the development states after the mechanism check. Its held-out run (`--part heldout_center`, FLTrust, anchor rule, CPQ-R and CPQ-B, clean and rotated sign-flip, same seeds) is confirmatory for:

- **H5 (centered trust).** In the clean condition, for each of FLTrust and the anchor rule, the lowest state weight under the centered cosine is higher than under the standard cosine (paired by seed), and pooled F1 is not lower by more than 0.005.
- **H6 (robustness kept).** Under sign-flip, centered FLTrust and the centered anchor rule keep pooled F1 within 0.01 of their clean value.
