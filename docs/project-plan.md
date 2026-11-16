# Research Programme and Milestone Record

This document records the order in which the work was built, the methodological commitments it was built under, and the decisions that were deferred or abandoned. See [research-question.md](research-question.md) for the question, [architecture.md](architecture.md) for the system design, and [rl-formulation.md](rl-formulation.md) for the RL specification and its divergences from the as-built system.

## 1. Methodological commitments

These were adopted at the outset and constrain what may be claimed. They are stated here because several of the findings elsewhere in this documentation exist only because these rules were followed — and, in the case of the provenance failure described in [evaluation.md §2](evaluation.md#2-provenance-and-status-of-the-evidence), because one of them was not enforced by tooling.

1. Build incrementally, one milestone at a time, validating each before starting the next.
2. Keep every interface testable in isolation.
3. Keep the RL research layer runnable without ROS2.
4. Prefer the smallest implementation satisfying the milestone; create no speculative generality and no placeholder modules.
5. Record technical decisions in documentation rather than leaving them implicit in code.
6. Flag physically invalid or research-invalidating assumptions rather than implementing them silently.
7. Treat no simulator or library capability as working until it has been exercised in this environment.
8. Describe no experiment as having run, and no baseline as working, until it has been executed and its output inspected.
9. Invent no performance numbers.

## 2. Milestone record

| Milestone | Deliverable | Outcome |
|---|---|---|
| M0 | Project bootstrap | **Complete.** Packaging, linting, formatting, type checking, tests, CI. |
| M1 | Simulation: marine environment, vessel, manipulator | **Complete.** Gazebo Harmonic 8.11.0 and VRX (`jazzy`) build and launch; WAM-V spawns in the `sydney_regatta` world with full ROS2 state, sensor, thruster and disturbance topics. A Franka Panda is mounted via a single fixed joint, with `ros2_control`/`gz_ros2_control` wired and commanded joint motion converging to target within 1e-9 rad. |
| M2 | ROS2 state pipeline | **Complete.** `marine_state_estimator` publishes vessel odometry, filtered arm joint state, and vessel-relative end-effector pose. TF2 tree mapped by inspection; see [ros2.md](ros2.md). |
| M3 | Conventional grasping, static target | **Complete.** MoveIt2 planning config for the vessel-mounted arm; IK, plan, execute and scripted attach validated against the live simulation, with rigidity confirmed by a second independent motion rather than inferred from proximity. |
| M4 | Dynamic grasping environment, fast backend | **Complete.** `src/opportunity_rl/sim/` implements `SimulationInterface` via `FastSimulator`: a Gerstner/Pierson-Moskowitz vessel disturbance ported from VRX, an independently moving target, a reachable-workspace manipulator proxy, and one shared episode/feasibility implementation. |
| M5 | RL environment | **Complete.** `DynamicGraspEnv` with the fixed 47-dimensional observation schema, the five-term reward, and the opportunity append slot. Confounds 1, 2, 6 and 7 realised structurally and unit-tested. |
| M6 | Standard RL baseline (B2) | **Complete.** A self-contained clipped PPO with GAE. Two substantive problems were found by inspecting the trained policy rather than by unit tests: the default disturbance configuration was too easy to demonstrate learning, and a wave startup ramp made step 1 a guaranteed opportunity, which the policy exploited. |
| M7 | Reactive benchmarks (B0, B1) and evaluation matrix | **Complete.** The load-bearing finding was that learned reactive B2 (+0.70) nearly matched perfect reactive B0 (+0.74), leaving almost no headroom for anticipation — which motivated the anticipation-required task variant rather than testing H1 on a task that could not discriminate. |
| M8 | Opportunity mechanism | **Complete.** Analytic second-order `OpportunityEstimator`, a committed multi-step grasp mode, and the mild/medium/severe severity ladder. |
| M9 | Opportunity-aware RL (B4) | **Complete.** B4 is B2 plus the appended opportunity block, with confounds enforced by construction. |
| M10 | Robustness sweep | **Partially complete.** The severity sweep ran across 3 severities, budgets and seeds. The distribution-shift split required by H3 was not built. |
| M11 | Research evaluation | **Complete but provisional.** [evaluation.md](evaluation.md) reports the design, statistics and threats to validity. The results are not citable as they stand; see §4 below. |

## 3. Decisions deferred, and what became of them

Several components named in early planning were deliberately not created until a milestone needed them, on the principle that an empty module is scaffolding with nothing to validate against. Recording which were eventually built, and which were not, prevents the plan from reading as a description of the system:

- **Built as planned:** `configs/` (at M4), `experiments/` and `scripts/` (at M6), `results/` (after the first run), and four ROS2 packages — `wamv_manipulator_description`, `marine_state_estimator`, `wamv_manipulator_moveit_config`, `marine_manipulation`.
- **Never built:** the ROS2 packages `marine_sim_bringup`, `marine_rl_policy` and `marine_interfaces`. The first two were obviated by launching through VRX's own tooling and by training entirely on the fast backend; the third was unnecessary because every field needed at the ROS2 boundary is expressible with standard `nav_msgs`, `sensor_msgs` and `geometry_msgs` types.
- **Never written:** `docs/experiments.md` and `docs/research-log.md`, both planned at M6. Their intended content is covered by [implementation-status.md](implementation-status.md) and the per-run metadata under `results/`.
- **Resolved during the work:** the manipulator selection (Franka Panda, at M1), the observation dimensionality (47, fixed at M5), and the algorithm choice (PPO over SAC, confirmed at M6 on the grounds that the action space is discrete).
- **Abandoned:** the staged curriculum specified in [rl-formulation.md §11](rl-formulation.md#11-curriculum), which was replaced by three severity conditions used in parallel rather than as a progression; and B3, whose construction was conditional on evidence of temporal dependence that was never gathered.

## 4. Status and what remains

All eleven milestones were built and validated on the fast backend, and the research question has an executed answer. That answer is nuanced rather than clean: opportunity-awareness accelerates learning without raising the performance ceiling.

Three things stand between the current state and a defensible result:

1. **Regenerate the experiments** against the corrected code, with provenance recorded as a real revision hash. Every existing measurement predates four correctness fixes and records `"git_commit": "HEAD"`.
2. **Resolve H2**, whose verdict differs between the 3-seed and 8-seed samples and is currently undetermined.
3. **Commit the statistical analysis as a script**, so that the reported figures can be reproduced by running the repository rather than recomputed by hand.

Beyond that, the open research work is: running H4 at an adequate seed count (the oracle timing reference and `experiments/oracle_regret.py` now exist), the distribution-shift split (H3), temporal RL (B3), and training against the ROS2/Gazebo backend.

## 5. Non-goals

Restated so they are not lost in implementation detail: no novel low-level controller, no novel inverse kinematics, no custom physics engine, no end-to-end camera-to-torque RL, no multi-agent RL, no sim-to-real transfer, no hardware integration, and no custom visualisation application. These may be future directions but are outside this programme.
