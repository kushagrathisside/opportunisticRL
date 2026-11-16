# Opportunity-Aware Reinforcement Learning for Dynamic Grasping under Marine Disturbance

[![CI](https://github.com/kushagrathisside/opportunisticRL/actions/workflows/ci.yml/badge.svg)](https://github.com/kushagrathisside/opportunisticRL/actions/workflows/ci.yml)

A controlled study of whether giving a reinforcement-learning policy an explicit representation of *future interaction opportunity* improves its grasp-timing decisions, relative to an otherwise-identical policy that reacts only to the current instant.

The setting is a manipulator mounted on a wave-disturbed vessel, grasping an independently moving target. Marine simulation is infrastructure, not the object of study: wave- and current-driven disturbance is a physically natural, configurable source of the temporal structure the research question concerns.

## Headline finding

Opportunity-awareness makes the timing policy **learn faster, but not end up better.**

| Regime | Result | Statistic |
|---|---|---|
| Low training budget (10k–20k steps) | Opportunity-aware policy (B4) beats reactive (B2) | 21 / 24 independent units, p = 2.8 × 10⁻⁴, mean Δreturn +0.227 |
| High training budget (40k–80k steps) | No detectable difference | 14 / 24 units, p = 0.54, mean Δreturn −0.003 |

The most defensible reading is that the base observation already contains relative velocity, from which a *learned* reactive policy can infer future opportunity given enough data. The explicit signal therefore acts as an inductive bias and a head start rather than raising the performance ceiling.

> **These results are provisional and should not be cited as they stand.** Every measurement predates four correctness fixes, records no usable revision hash, and is demonstrably not reproducible — re-running the same seeds produces different curves. [docs/evaluation.md §2](docs/evaluation.md#2-provenance-and-status-of-the-evidence) states the problem precisely and §9 states what regeneration requires.

## Hypothesis status

| ID | Hypothesis | Status |
|---|---|---|
| H1 | Opportunity representation improves grasp decision-making | **Split.** Supported for sample efficiency; not supported for final performance. |
| H2 | The benefit grows with disturbance magnitude | **Undetermined.** A 3-seed sample and an 8-seed sample give opposite orderings. |
| H3 | Opportunity-awareness generalises better under distribution shift | **Not tested.** The train/test split was never built. |
| H4 | How much of the gap to a future-aware oracle is closed | **Not yet tested at scale.** The oracle and `experiments/oracle_regret.py` now exist; the experiment has not been run at an adequate seed count. |

Full statements, pre-registered support and rejection criteria, and evidence: [docs/research-question.md](docs/research-question.md) and [docs/evaluation.md](docs/evaluation.md).

## What is here, and what is not

**Built and validated:** a Gazebo Harmonic + VRX simulation with a Franka Panda mounted on a WAM-V and a validated conventional grasp against the live stack; a ROS2 state-estimation pipeline; a fast pure-Python `SimulationInterface` backend with a source-faithful Gerstner/Pierson-Moskowitz wave disturbance and an independently moving target; a Gymnasium `DynamicGraspEnv` with a fixed 47-dimensional observation schema and a five-term reward; the baseline ladder B0, B1, B2 and B4; a leakage-safe analytic `OpportunityEstimator`; and the matched H1 comparison across seeds, budgets and severities.

**Not built:** B3 (temporal RL), the distribution-shift split, and the ROS2/Gazebo training backend — no policy has ever been trained or evaluated through ROS2. Specification-to-implementation divergences, including a feasibility predicate simpler than the one specified, are collected in [docs/rl-formulation.md §9](docs/rl-formulation.md#9-divergences-from-specification-collected).

## Documentation

[docs/README.md](docs/README.md) indexes all documentation by evidential role — specification, record, or review. For assessing the research, read [docs/research-question.md](docs/research-question.md) then [docs/evaluation.md](docs/evaluation.md).

[docs/dashboard.html](docs/dashboard.html) is a single self-contained page — no server or build step — carrying the milestone tracker, confound register, RL specification, environment telemetry, and an in-browser toy simulator of the opportunity-window mechanism. It is a manually regenerated snapshot rather than a live view.

## Installation

The Python research layer requires no ROS2:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # tooling and tests
pip install -e ".[rl]"       # torch, gymnasium, numpy — needed to train
pip install -e ".[viz]"      # matplotlib — needed only for figures
```

The ROS2 workspace (Jazzy) is built separately; see [ros2_ws/README.md](ros2_ws/README.md). Use **system Python**, not a conda environment, for anything ROS2 — the two Python environments are kept separate by design, since the RL training path does not require ROS2 at all.

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws && colcon build --symlink-install
```

## Running the simulation

```bash
scripts/launch_vrx_sim.sh headless:=True world:=sydney_regatta \
    urdf:=$(pwd)/install/wamv_manipulator_description/share/wamv_manipulator_description/urdf/wamv_with_arm.urdf.xacro
```

First launch downloads roughly 159 MiB of world assets through Gazebo's Fuel service and may take several minutes; subsequent launches use the local cache. Setup details, exact versions, and every environment defect encountered are in [docs/environment.md](docs/environment.md), alongside captured media of the running simulation.

## Reproducing the experiments

```bash
# Baseline matrix. Pass the same --grasp-duration-steps the policies were trained at;
# the default of 1 measures a different task and is not comparable to the H1 result.
python3 experiments/evaluation_matrix.py --disturbance moving_target --grasp-duration-steps 3

# Train the reactive baseline. Use --disturbance moving_target: the default
# sydney_regatta_default configuration is deliberately easy (the target is always
# graspable, so timing is irrelevant).
python3 experiments/train_b2.py --disturbance moving_target --seed 0

# H4: timing regret against the oracle timing reference, and the share of
# B2's gap to the oracle that B4 closes (trains B2 and B4 per seed).
python3 experiments/oracle_regret.py --disturbance moving_target \
    --grasp-duration-steps 3 --total-steps 80000 --seeds 0 1 2

# The H1 sweep behind the reported figure.
python3 experiments/learning_curve_h1.py \
    --disturbances moving_target_mild moving_target moving_target_severe \
    --grasp-duration-steps 3 --seeds 0 1 2
```

Each training run writes `results/<run>/metadata.json` capturing seed, git revision, resolved configuration, total environment steps and evaluation results, alongside a checkpoint. `results/` is gitignored. Training and evaluation never share a seed; evaluation uses `seed + 10_000`.

```bash
# Sign tests, per-cell table and per-severity (H2) breakdown from a sweep's output.
python3 experiments/analyze_h1.py results/h1_learning_curve/curve_data.json
```

**Note on reproduction:** the statistics currently reported in the documentation were computed by ad hoc analysis before `experiments/analyze_h1.py` existed, and the `curve_data.json` files they came from are not in the repository. The script implements the independent-unit scheme from [docs/evaluation.md §4](docs/evaluation.md#4-statistical-analysis). Re-running the sweep and then the script is how the reported numbers should be regenerated.

## Known limitations

1. **All results are from the fast kinematic backend.** The ROS2/Gazebo backend implements the same interface but was never trained against. The timing structure the hypothesis concerns is represented faithfully; low-level manipulation is not, by design.
2. **The feasibility predicate is simpler than specified** — distance and relative speed only, with no grasp-geometry, orientation or angular-velocity terms, and a zero end-effector-velocity proxy. Both policies face the identical predicate, so the comparison stands, but the generality of any claim is bounded accordingly.
3. **The collision penalty never fires**, because the fast backend has no contact model. The pathway is wired and inert.
4. **Success rate is saturated and uninformative** on these tasks; return is the discriminating metric throughout.
5. **MoveIt2 planning is exercised only by the scripted M3 grasp**, never by an RL policy.
6. **DART cannot build collision geometry** from the Panda's stock meshes and does not enforce the gripper's mimic joint — pre-existing engine limitations, relevant once real grasp contact is implemented.

## Exploratory visualisation

`scripts/visualize_opportunity_concept.py` is a standalone numpy and matplotlib illustration of the research question — a disturbed vessel, a drifting target, a good-opportunity window, and a reactive-versus-lookahead comparison. It is **not** the environment, the estimator, or any part of the experimental pipeline, and nothing in it should be cited as a result.

```bash
pip install -e ".[viz]"
python3 scripts/visualize_opportunity_concept.py            # static figure
python3 scripts/visualize_opportunity_concept.py --video    # animation (requires ffmpeg)
```

## Repository layout

```text
docs/                 research question, architecture, RL specification, evaluation, reviews
src/opportunity_rl/   RL research layer (ROS2-independent)
  sim/                SimulationInterface, fast backend, wave disturbance, episodes
  env/                DynamicGraspEnv, observation schema, reward
  algorithms/         PPO, heuristics, random policy
  opportunity/        OpportunityEstimator
ros2_ws/src/          ROS2 packages (description, state estimator, MoveIt2 config, grasp pipeline)
tests/                unit tests for src/opportunity_rl
configs/disturbances/ named disturbance conditions, one YAML per condition
experiments/          training, comparison and evaluation-matrix scripts
scripts/              diagnostics and the exploratory visualiser
results/              run artefacts (gitignored)
```

## Development

```bash
ruff check . && ruff format --check .
mypy src/opportunity_rl
pytest
```

**Environment note:** if your shell sources ROS2's `setup.bash`, `PYTHONPATH` will contain ROS2's `site-packages`, which leaks `launch_testing` into pytest's plugin autodiscovery and fails before any test runs. Run `unset PYTHONPATH` first, or keep a separate shell for `src/opportunity_rl` work. This does not affect CI, which has no ROS2 installed.

## License

MIT — see [LICENSE](LICENSE).
