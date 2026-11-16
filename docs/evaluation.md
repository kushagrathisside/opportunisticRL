# Research Evaluation: The H1 Controlled Experiment

This document reports the controlled test of **H1** — *does explicitly representing future interaction opportunity improve RL decision-making for dynamic grasping under disturbance, relative to reacting to the current instant?* — together with the severity analysis bearing on **H2**. It states the experimental design, the statistical analysis, the findings, and the conditions under which those findings may legitimately be cited.

> **Status of the evidence: provisional, pending regeneration.** Every measurement described here was produced by a revision of the code that has since been found to contain four defects affecting this computation, and none of the runs is reproducible from its recorded metadata. The direction of the principal effect is stable across two independent samples, but no number below should be quoted as a settled result. Section 2 states the problem precisely; Section 9 states what must be done before these results are citable.

## 1. Summary of findings

1. **Sample efficiency (H1).** At low training budgets (10k–20k environment steps) the opportunity-aware policy (B4) attains higher return than the matched reactive policy (B2). The effect is large and consistent: 21 of 24 independent seed × severity units favour B4 (exact sign test, p = 2.8 × 10⁻⁴) on the larger of the two available samples.
2. **Final performance (H1).** By 40k–80k steps the two policies are statistically indistinguishable (14 of 24 units, p = 0.54; mean Δreturn = −0.003). The explicit opportunity signal does not raise the asymptotic performance ceiling.
3. **Severity dependence (H2).** The two available samples disagree, and the disagreement is not decidable from the data in hand. The 3-seed sample gives a non-monotone ordering and was previously reported as rejecting H2; the 8-seed sample gives an ordering that increases monotonically with disturbance severity and is consistent with H2. **H2 is therefore reopened as undetermined**, not rejected.
4. **Interpretation.** Taken together, (1) and (2) indicate that the value of an explicit future-opportunity representation in this setting is as an inductive bias that accelerates learning, rather than as a source of higher converged performance — plausibly because the base observation already contains relative velocity, from which a learned reactive policy can approximate future opportunity given sufficient data.

## 2. Provenance and status of the evidence

Three distinct problems constrain how strongly these results can be stated. They are recorded here rather than in a footnote because they bear directly on the interpretation of every number in Section 6.

**The measurements predate four correctness fixes.** All run artefacts under `results/` were produced before the following defects were corrected: a vessel/world frame mismatch in the published target state; the same mismatch inside the opportunity estimator (which fed B4's live policy input); unseeded training-episode resets; and a stale observation carried across the interleaved train/evaluate boundary, which affected the learning-curve script at every budget. The frame errors were measured to flip no grasp decision under the shipped configurations, and the reported direction of the H1 effect is not expected to change, but the results have not been regenerated against the corrected code.

**The runs are not reproducible from their recorded metadata.** Every `metadata.json` under `results/` records `"git_commit": "HEAD"` rather than a revision hash, because the provenance helper did not check git's return code. The revision that produced each number is therefore unrecoverable.

**Non-reproducibility is demonstrated, not merely suspected.** Seeds 0–2 were run twice under nominally identical settings (`results/h1_severity_curves/` and `results/h1_severity_curves_n8/`). All 18 comparable series differ. In one instance the divergence is large enough to change the qualitative reading of a cell: at mild severity and a 10k budget, B2 scores −1.220 in the first run (a policy that has not yet learned) and +0.526 in the second (a policy that has). This is the expected consequence of the unseeded-reset defect and is the clearest available evidence that the low-budget estimates are unstable at small seed counts.

Because `results/` is gitignored, these artefacts do not ship with the repository; the figure in Section 6 and the tables below are the only published record of them.

## 3. Experimental design

The comparison is **B2 (reactive RL)** against **B4 (opportunity-aware RL)**, matched in every respect except the variable under test:

- **Algorithm and capacity.** The same clipped-PPO implementation (`src/opportunity_rl/algorithms/ppo.py`), the same hyperparameters, the same network capacity.
- **Task.** The same environment, disturbance configuration, reward, decision rate, and episode structure.
- **Budget and seeds.** The same training-step budgets and the same seeds, with evaluation on held-out seeds that never overlap training (`seed + 10_000`).
- **Observation.** The same *base* observation. B4's only difference is an `OpportunityEstimator` output appended as a separate block, never interleaved with the base fields.

These correspond to confound-register items 1, 2, 6 and 7 (see [architecture.md](architecture.md#4-confound-register)), and each is enforced structurally rather than by convention: the two policies are constructed through one code path with and without the appended block, and unit tests assert that the base observation blocks are byte-identical (`tests/opportunity/test_b4_env_integration.py`). The estimator is leakage-safe by construction — it consumes only current and one-step-past state and extrapolates, never the simulator's true future trajectory (`tests/opportunity/test_estimator.py`).

**Task.** The `moving_target` family places a manipulator on a wave-disturbed vessel and an independently bobbing target that is only intermittently graspable, with a good-opportunity window that opens and closes as the target decelerates at its oscillation turning points. Grasping is a committed multi-step manoeuvre (`grasp_duration_steps = 3`) that fails if the window closes before completion, so committing to a grasp the window will not sustain is a real and penalised error — the class of decision for which anticipating the opportunity trajectory should, if H1 holds, be advantageous. The severity ladder is defined by the measured fraction of time a good opportunity exists: **mild 0.55, medium 0.33, severe 0.21**.

## 4. Statistical analysis

The unit of analysis is the matched B4-minus-B2 difference in mean evaluation return. Two aggregation schemes are possible, and they are not equivalent:

- **Pooled over budgets.** Treating each (severity, seed, budget) triple as one observation yields 18 pairs at n = 3 and 48 at n = 8. This is the scheme used in earlier versions of this document.
- **Independent units.** Because the learning-curve script trains **one** PPO instance incrementally and evaluates it at successive budgets (`experiments/learning_curve_h1.py`), the budgets within a given (severity, seed) are repeated measures on a single training run, not independent samples. Pooling them overstates the effective sample size and yields an optimistic p-value.

**The independent-unit scheme is the defensible one**, and is used for the headline statistics below: within each (severity, seed) unit the per-budget differences are averaged, and an exact two-sided sign test is applied across units. Pooled figures are retained alongside for continuity with the earlier reporting. The qualitative conclusion is unchanged under either scheme, which is itself worth stating — the correction to the analysis does not rescue or destroy the finding.

No correction for multiple comparisons is applied, because only two pre-specified regimes (low and high budget) are tested. The sign test is used rather than a t-test because the per-unit differences are neither obviously normal nor homoscedastic across severities, and because the direction of the effect, rather than its precise magnitude, is what H1 predicts.

## 5. Statistical results

| Sample | Regime | Independent units favouring B4 | p (exact sign test) | Mean Δreturn | Pooled (for continuity) |
|---|---|---|---|---|---|
| n = 8 | Low budget (10k, 20k) | **21 / 24** | **2.8 × 10⁻⁴** | **+0.227** | 40/48, p = 3.3 × 10⁻⁶ |
| n = 8 | High budget (40k, 80k) | 14 / 24 | 0.54 | −0.003 | 20/48, p = 0.31 |
| n = 3 | Low budget (10k, 20k) | 9 / 9 | 3.9 × 10⁻³ | +0.380 | 17/18, p = 1.5 × 10⁻⁴ |
| n = 3 | High budget (40k, 80k) | 5 / 9 | 1.0 | +0.002 | 8/18, p = 0.81 |

The n = 8 sample is the larger and is preferred; the n = 3 sample is reported because it is the one the published figure depicts.

## 6. Learning curves and per-cell detail

The figure below is generated from the **n = 3** sample. A corresponding n = 8 figure exists at `results/h1_severity_curves_n8/learning_curve.png` but has not been published, and regenerating this figure from the larger sample is part of the outstanding work in Section 9.

![H1 learning curves, three seeds](images/h1_learning_curve.png)

**Per-cell mean return, B2 / B4 (Δ = B4 − B2; parenthesised count is seeds favouring B4), n = 8:**

| Severity | 10k | 20k | 40k | 80k |
|---|---|---|---|---|
| mild | −0.34 / −0.77 (Δ −0.43, 4/8) | +0.63 / +0.72 (Δ +0.08, 7/8) | +0.77 / +0.72 (Δ −0.04, 0/8) | +0.75 / +0.77 (Δ +0.01, 4/8) |
| medium | −1.22 / −0.77 (Δ +0.45, 8/8) | +0.30 / +0.61 (Δ +0.31, 8/8) | +0.63 / +0.65 (Δ +0.02, 7/8) | +0.67 / +0.68 (Δ +0.01, 6/8) |
| severe | −1.22 / −0.78 (Δ +0.43, 6/8) | +0.02 / +0.53 (Δ +0.51, 7/8) | +0.56 / +0.55 (Δ −0.01, 0/8) | +0.57 / +0.56 (Δ −0.01, 3/8) |

**Per-cell mean return, n = 3** (the sample underlying the published figure):

| Severity | 10k | 20k | 40k | 80k |
|---|---|---|---|---|
| mild | −1.22 / −0.58 (Δ +0.64, 3/3) | +0.61 / +0.69 (Δ +0.09, 3/3) | +0.77 / +0.73 (Δ −0.04, 1/3) | +0.72 / +0.76 (Δ +0.04, 2/3) |
| medium | −1.22 / −0.66 (Δ +0.55, 3/3) | −0.03 / +0.61 (Δ +0.64, 3/3) | +0.66 / +0.66 (Δ +0.01, 2/3) | +0.67 / +0.68 (Δ +0.01, 2/3) |
| severe | −1.22 / −0.87 (Δ +0.34, 3/3) | +0.51 / +0.53 (Δ +0.02, 2/3) | +0.56 / +0.55 (Δ −0.01, 0/3) | +0.56 / +0.57 (Δ +0.01, 1/3) |

## 7. H1: what the evidence supports

**Sample efficiency — supported.** At low budgets B4 attains higher return than B2 in 21 of 24 independent units (p = 2.8 × 10⁻⁴, mean Δ = +0.227). The learning curves show B4 reaching a competent policy while B2 remains near its timeout floor of approximately −1.2. The explicit opportunity signal materially accelerates acquisition of the timing decision. This is consistent with the expectation recorded in [related-work-review.md](related-work-review.md) — that residual- and opportunity-style signals are less data-hungry — which was written before the experiment was run rather than fitted to it afterwards.

**Final performance — not supported.** By 40k–80k steps the two policies converge to statistically indistinguishable return (14 of 24 units, p = 0.54, mean Δ = −0.003). The most defensible explanation, foreshadowed by the M7 result in which the learned reactive policy B2 already nearly matched the perfect-reactive heuristic B0, is that the base observation contains relative velocity, from which a learned reactive policy can infer future opportunity given enough data. The explicit signal is therefore largely redundant at convergence and valuable principally as a head start.

**Success rate is not the discriminating metric.** Episodes are long enough that even a random policy eventually grasps, so success rate saturates near 1.0 for every policy evaluated. Return, which also charges for failed attempts and elapsed time, is the metric used throughout. This was identified at M6 and shaped the evaluation design rather than being discovered during analysis.

## 8. H2: severity dependence, reopened

H2 asks whether the benefit of opportunity-awareness grows with disturbance magnitude. The two samples give opposite answers, and the discrepancy is instructive.

| Low-budget mean Δreturn by severity | mild | medium | severe | Ordering |
|---|---|---|---|---|
| n = 3 (previously reported) | +0.365 | +0.595 | +0.182 | non-monotone |
| n = 8 | −0.174 | +0.380 | +0.474 | increases with severity |

On the 3-seed sample the advantage is largest at mild and medium and smallest at severe, which does not support H2, and this document previously recorded H2 as *rejected in this setting*. On the 8-seed sample the advantage is absent at mild (5/8 seeds, p = 0.73), and strong and unanimous at medium (8/8, p = 7.8 × 10⁻³) and severe (8/8, p = 7.8 × 10⁻³), giving an ordering that increases monotonically with severity — the pattern H2 predicts.

The 8-seed ordering also admits a coherent mechanism: mild severity leaves a good opportunity available 55% of the time, so a reactive policy learns adequate timing quickly without assistance and there is little headroom for an anticipatory signal; as opportunities become rarer the timing decision becomes harder and the signal becomes more valuable. That this mechanism is plausible does not make it established.

**Conclusion: H2 is undetermined.** The earlier rejection rested on a 3-seed sample now known to be unstable at exactly the low budgets the comparison depends on, and it is contradicted by the larger sample. Neither verdict should be asserted until the experiment is regenerated against the corrected code with an adequate seed count.

## 9. Threats to validity, and conditions for citing these results

**Provenance.** As set out in Section 2, the measurements predate four correctness fixes, record no usable revision hash, and are demonstrably not reproducible. This is the dominant limitation.

**Statistical power.** The 3-seed sample is underpowered for the low-budget regime, where between-seed variance is largest; the H2 reversal in Section 8 is a direct demonstration. The 8-seed sample is adequate for the direction of the H1 sample-efficiency effect but remains thin for per-severity claims, each of which rests on 8 units.

**Analysis not scripted (partly addressed).** The statistics in Sections 5 and 8 were computed from `curve_data.json` by ad hoc analysis, not by a committed script. That analysis is now committed as `experiments/analyze_h1.py` (logic in `src/opportunity_rl/h1_analysis.py`, unit-tested, including a check that its sign test reproduces the p-values in Section 5 from the reported counts). The numbers in this document have **not** yet been regenerated with it, because the original `curve_data.json` files are not in the repository and the sweep itself has to be re-run against the corrected code first.

**Fast backend only.** All results are from the pure-Python kinematic backend, not the ROS2/Gazebo backend, which implements the same interface but was never trained against. The wave disturbance is a source-faithful port of VRX's model (see [wave-simulation-literature-review.md](wave-simulation-literature-review.md)), but the manipulator is a reachable-workspace proxy rather than a 7-DOF dynamics simulation. The timing structure the hypothesis concerns is represented faithfully; the low-level manipulation is not, by design.

**The estimator is analytic, not learned.** It is the deterministic second-order extrapolation specified as the first opportunity representation. A learned trajectory predictor might raise the ceiling where this one does not, which is the natural next experiment motivated by the null final-performance result.

**Before any of these numbers is cited**, the following are required: (1) regenerate the full sweep against the corrected code, with provenance recorded as a real revision hash; (2) use at least the 8-seed count, and preferably more for the per-severity H2 claim; (3) commit the analysis as a script so the reported statistics are reproducible; and (4) republish the figure from the sample actually being reported.

## 10. Outstanding work

The oracle timing reference required by **H4** has since been built (`algorithms/oracle.py`, run via `experiments/oracle_regret.py`), but no H4 result is reported here yet. The following were specified but not built, and are recorded as open rather than folded into the conclusions above: the train/test distribution split required by **H3**; the temporal-RL baseline **B3**; and training against the ROS2/Gazebo backend. See [research-question.md](research-question.md) for the per-hypothesis status and [project-plan.md](project-plan.md) for how each relates to the milestone sequence.
