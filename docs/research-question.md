# Research Question and Hypotheses

This document is the authoritative statement of the question this repository exists to answer, the hypotheses derived from it, and the criteria by which each is judged. Every other design decision — environment, reward, baselines, evaluation protocol — is subordinate to it. An implementation choice that would invalidate the comparison specified here is a defect regardless of whether the code executes correctly.

## 1. The question

> Does explicitly reasoning about future interaction opportunities improve RL decision-making for dynamic robotic grasping under stochastic disturbances, compared with reactive decision-making?

## 2. Motivation and choice of setting

A manipulator mounted on a disturbed platform, grasping an independently moving target, is a minimal setting in which *when* to act matters as much as *how*. A purely reactive policy can assess only the present instant: whether the target is reachable now. But grasp quality is a trajectory through time — a target may be approaching a favourable configuration, passing through one, or about to leave it. A policy given some representation of where that trajectory is heading can, in principle, wait for a better moment or commit before a good moment passes, rather than responding to the current instant alone.

Marine and USV simulation, together with ROS2, are the *setting* that produces this temporal structure cheaply: wave- and current-driven disturbance is a physically natural and configurable source of a moving reference frame and a moving target. They are infrastructure, not the object of study. The scientific claim is about decision-making under temporal structure, not about marine robotics or ROS2 integration.

## 3. Hypotheses and current status

All statuses below are **provisional**. The evidence supporting them predates four correctness fixes and is not reproducible from its recorded metadata; see [evaluation.md §2](evaluation.md#2-provenance-and-status-of-the-evidence). Any future change to a status must cite an executed run.

| ID | Hypothesis | Status |
|---|---|---|
| **H1** | A policy whose observation includes a representation of future interaction quality achieves higher grasp success and/or lower timing regret than an otherwise-identical policy without it | **Split by training budget.** Supported as a sample-efficiency effect; not supported as a final-performance effect. |
| **H2** | The benefit of opportunity-awareness increases with disturbance magnitude | **Undetermined.** Previously reported as rejected on a 3-seed sample; contradicted by an 8-seed sample. |
| **H3** | An opportunity-aware policy generalises better to disturbance regimes outside its training distribution | **Not tested.** Requires a train/test distribution split that was not built. |
| **H4** | How much of the achievable gap to a future-aware oracle does the opportunity-aware policy close? | **Not yet tested at scale.** The oracle timing reference (`algorithms/oracle.py`) and the experiment (`experiments/oracle_regret.py`) exist; no run at an adequate seed count has been reported. |

### H1 (primary)

An RL policy whose observation includes a representation of future interaction quality — an *opportunity* signal, specified in [rl-formulation.md](rl-formulation.md#4-future-opportunity-representation) — achieves higher grasp success rate and/or lower timing regret against an oracle than a policy with an otherwise-identical observation, network capacity, action space, reward, training budget, and disturbance distribution, but without the opportunity signal.

H1 is stated as a **controlled comparison**, not as an isolated claim about one policy's performance. An absolute figure such as "82% success rate" carries no evidential weight on its own; only the difference between the opportunity-aware and reactive conditions under matched conditions bears on H1.

Measured outcome, from matched B2-versus-B4 PPO across 8 seeds and 3 disturbance severities with held-out evaluation:

1. **Sample efficiency — supported.** At 10k–20k environment steps, B4 exceeds B2 in 21 of 24 independent seed × severity units (exact sign test, p = 2.8 × 10⁻⁴; mean Δreturn +0.227).
2. **Final performance — not supported.** By 40k–80k steps the two converge (14 of 24 units, p = 0.54; mean Δreturn −0.003). The opportunity signal does not raise the ceiling.
3. **Interpretation.** The base observation already includes relative velocity, from which a *learned* reactive policy can infer future opportunity given sufficient data. The explicit signal therefore acts as an inductive bias and a head start rather than as a source of higher asymptotic performance.

Success rate saturates on these tasks and does not discriminate between policies; return is the reported metric throughout. See [evaluation.md](evaluation.md) for the design, the statistical treatment, and the threats to validity.

### H2 (secondary)

Whether the benefit of opportunity-awareness grows with disturbance magnitude was tested via a mild/medium/severe severity ladder (good-opportunity fractions 0.55 / 0.33 / 0.21). The two available samples disagree: on 3 seeds the low-budget advantage is non-monotone in severity (+0.365 / +0.595 / +0.182), which does not support H2; on 8 seeds it increases monotonically (−0.174 / +0.380 / +0.474), which is the pattern H2 predicts. The earlier rejection rested on the smaller sample, which is now known to be unstable at precisely the low budgets the comparison depends on. **H2 is recorded as undetermined** until the experiment is regenerated against corrected code with an adequate seed count.

### H3 and H4 (open)

Neither has been tested. **H3** requires training on one severity and evaluating on another. **H4** requires an oracle with access to the true future, against which timing regret is measured. That oracle now exists (`algorithms/oracle.py`), and `experiments/oracle_regret.py` reports timing regret and the gap-closed ratio (B4 − B2) / (oracle − B2), so H4 needs only an adequately seeded run. Both are recorded here rather than silently dropped, and the environment and metrics already accommodate them: the opportunity representation is swappable by design, and the oracle is specified as an evaluation reference that is never used as a policy input.

## 4. Criteria for support and rejection

These criteria were fixed before the experiment was run and are restated unchanged:

- **Supports H1** — the opportunity-aware policy significantly outperforms the reactive policy, paired across seeds under matched training budget and disturbance distribution, on the primary metric or on timing regret, across a disturbance sweep, without leaking future ground truth (see [confound 7](architecture.md#4-confound-register)).
- **Rejects H1** — no statistically significant difference, or the reactive policy matches or exceeds the opportunity-aware policy under matched conditions.
- **Inconclusive**, and reported as such rather than folded into either verdict — any comparison in which a confound listed in the [confound register](architecture.md#4-confound-register) is present and uncorrected, or in which the provenance of the measurement cannot be established.

The third criterion currently applies to the provenance of all reported results, which is why every status in Section 3 is marked provisional.

## 5. Non-goals

This project does not claim novelty in marine robotics, ROS2, or MoveIt2 integration; these are reused off-the-shelf infrastructure. It does not aim to produce a deployable or hardware-validated grasping system, and it does not propose a new RL algorithm — the baselines use standard published methods. The only claimed contribution is an empirical answer, under a documented experimental protocol, to the question in Section 1.

## 6. Relation to prior work

H1's novelty claim has been tested against the closest prior work located, spanning marine surface-manipulator grasping, prediction-based grasp timing, RL for marine rescue, and general dynamic-grasping and handover literature; see [related-work-review.md](related-work-review.md). No located work combines (1) a separately-observable *future interaction quality* representation, (2) feeding a discrete commitment decision, and (3) isolated by a matched controlled comparison. Two caveats qualify that conclusion: the two closest papers — a hook-based aerial payload grasping paper using optimal-time trajectory optimisation, and an RL-for-sea-rescue paper — should be addressed explicitly in any eventual write-up rather than implicitly; and two papers in the closest physical setting (USV plus manipulator grasping) could not be obtained in full text and are flagged as that review's principal open risk.
