# RL Formulation: Specification and As-Built Reconciliation

This document is the formal specification for the RL layer — action space, observation space, reward, episode structure, opportunity representation, and the baseline ladder. It was written before any RL code existed, so that the implementation could be checked against a fixed specification rather than the specification being inferred from whatever code happened to be written.

It now serves a second purpose. Where the implementation diverges from the original specification, the divergence is recorded here in a **Divergence** note rather than left to be discovered by reading the source. Section 9 collects these. Readers assessing what was actually built, as opposed to what was planned, should read the divergences first.

See [research-question.md](research-question.md) for why this formulation exists and [architecture.md §4](architecture.md#4-confound-register) for the confounds it is designed to avoid.

## 1. Framework

**PyTorch**, with a small custom PPO rather than a third-party RL library. The rationale was that no JAX-specific performance requirement existed, correctness rather than throughput was primary, and a discrete four-action space is directly served by PPO's categorical head. The choice was to be revisited only if a concrete bottleneck were measured; none was.

## 2. Action space

Discrete, four actions, shared by every baseline:

| Value | Name | Meaning |
|---|---|---|
| 0 | `WAIT` | Hold current pose; no approach or grasp motion commanded. |
| 1 | `APPROACH` | Command the manipulation manager toward the current grasp-candidate pose. |
| 2 | `ATTEMPT_GRASP` | Commit to a grasp execution via the conventional manipulation stack. |
| 3 | `ABORT` | Cancel an in-progress approach or grasp and return to a safe standby pose. |

**ABORT during a committed grasp.** With `grasp_duration_steps > 1`, a grasp is a multi-step maneuver during which `WAIT`, `APPROACH` and `ATTEMPT_GRASP` are ignored. `ABORT` cancels it: no grasp, no failure penalty, and the arm returns to standby. The window is checked *before* the abort, so once the window has closed the grasp has already failed and `ABORT` cannot erase that. Aborting therefore pays off only when the policy foresees the window closing while it is still open, which keeps the abort decision anticipatory rather than reactive. Earlier revisions ignored `ABORT` mid-maneuver as well, so results from before this change were produced with a narrower action semantics.

A fifth action (`REPOSITION`) and continuous parameters such as a target-relative offset were deferred until the four-action formulation could be shown insufficient by experiment. No such demonstration was produced, so the action space remains as specified.

Actions are issued at a fixed **decision period**, a configuration value rather than a per-baseline constant, which decouples the RL decision rate from the underlying controller's rate and guarantees confound 2 by construction.

## 3. Observation space

A flat, normalised vector with a fixed schema. The base dimension is **47** — the sum of the field widths below, computed from the schema rather than hard-coded. B4 appends a 4-dimensional opportunity block, giving it a 51-dimensional observation; B2's is the 47-dimensional base alone.

| Field | Width | Units and frame |
|---|---|---|
| `relative_target_position` | 3 | metres, target relative to end-effector |
| `relative_target_orientation` | 4 | quaternion, target relative to end-effector |
| `relative_target_linear_vel` | 3 | m/s |
| `relative_target_angular_vel` | 3 | rad/s |
| `ee_pose` | 7 | position (3) + quaternion (4), vessel-base frame |
| `joint_state` | 14 | position and velocity for 7 arm joints |
| `vessel_linear_vel` | 3 | m/s, world frame |
| `vessel_angular_vel` | 3 | rad/s |
| `distance_to_grasp_region` | 1 | metres |
| `orientation_error` | 1 | radians |
| `relative_approach_velocity` | 1 | m/s, closing speed along approach axis |
| `task_state` | 4 | one-hot over the four actions |

Three rules bind the implementation:

1. **Normalisation.** Every field is divided through by a characteristic scale before a policy sees it, using constants held in a `NormConfig` dataclass rather than inline in the conversion body. The raw units above are pre-normalisation definitions.
2. **Purity.** Conversion from simulator state to this vector is a deterministic function with no hidden simulator handle reachable from policy code.
3. **Separation.** The reactive and opportunity-aware policies read the *same* base schema. The opportunity-aware policy additionally receives the estimator's output appended as a separate block, never mixed into the base fields, so confound 1 is checkable by inspecting one field list rather than diffing two parallel schemas.

The optional `pose_uncertainty` and `velocity_uncertainty` fields are represented in the schema as zero-filled blocks, disabled by default. Enabling them is an explicit configuration change rather than a silent default. They were never enabled.

## 4. Future opportunity representation

The specification fixes an interface rather than a formula, admitting three representation shapes without an environment rewrite: a scalar opportunity score at the current instant; a future opportunity vector over a fixed horizon; or a probability of successful grasp at each of several horizons.

**As built:** the second shape — a future-opportunity vector `[O(t+d₁), …, O(t+d₄)]` at horizons **0.5, 1.0, 1.5 and 2.0 seconds**, giving a 4-dimensional appended block. Each element is a smooth analogue of the feasibility predicate evaluated at an extrapolated future state.

The estimator is **second-order**: it uses relative position, velocity, and a finite-difference acceleration derived from one step of velocity history. This matters mechanistically — good windows arise from the target *decelerating* at wave turning points, and a constant-velocity predictor is structurally blind to deceleration. A unit test confirms the second-order estimator's peak future opportunity exceeds a first-order estimator's by more than 0.2 on a decelerating approach.

A learned opportunity model, such as a trajectory predictor, was specified as a later extension contingent on the analytic version proving a bottleneck. It was not built, and the null final-performance result in [evaluation.md §7](evaluation.md#7-h1-what-the-evidence-supports) is the clearest motivation for attempting it.

**Future-information leakage rule (binding):**

| Input | Status |
|---|---|
| Current observation | Permitted |
| Prediction derived from current and past state | Permitted |
| Short-horizon *estimated* trajectory (extrapolated) | Permitted, provided it is derived rather than copied from simulator future state |
| True future simulator trajectory | **Prohibited** as a policy input. Reserved for the oracle reference, which is never used as a policy. |

The estimator is stateful — it holds one past velocity — and is reset by the environment each episode so that nothing crosses an episode boundary.

## 5. Grasp opportunity and feasibility

The specification defines a layered predicate, shared by every baseline that needs a feasibility notion so that heuristic behaviour and the opportunity signal derive from one definition rather than three:

```text
reachable            := target within manipulator workspace bound
graspable            := reachable AND grasp geometry valid (approach axis, clearance)
good_opportunity     := graspable AND orientation_error < θ_max
                                  AND relative_velocity < v_max
                                  AND relative_angular_velocity < ω_max
successful_execution := realized outcome from the conventional grasp controller,
                        reported after ATTEMPT_GRASP — not assumed from good_opportunity
```

> **Divergence.** The implemented predicate (`sim/episode.py:evaluate_feasibility`) is the reduced form
> `reachable := distance < workspace_radius_m` and `good_opportunity := reachable AND relative_speed < max_relative_speed_mps`.
> The intermediate `graspable` stage is absent, and neither `orientation_error` nor `relative_angular_velocity` participates. A second simplification compounds this: end-effector velocity is not tracked separately, so the "relative" velocity is the target's velocity in the vessel frame against a zero end-effector-velocity proxy. The task is therefore a **distance-and-speed timing problem**, not the full geometric grasp-feasibility problem the specification describes. This does not invalidate the H1 comparison, which turns on whether an anticipatory signal helps time a committed decision, and both policies face the identical predicate. It does bound the generality of any claim: the result concerns timing under a simplified feasibility model.

The distinction between `good_opportunity` and `successful_execution` is retained in the design: a good opportunity does not imply a successful grasp, which is what separates "did the policy pick a good moment" from "did the controller execute correctly."

## 6. Reward

```text
R_t = R_success · 1[success] − λ_f·C_failure − λ_c·C_collision − λ_t·C_time − λ_m·C_motion
```

As-built defaults: `R_success = 1.0`, `λ_f = 0.5`, `λ_c = 1.0`, `λ_t = 0.01` per step, `λ_m = 0.001`.

- One implementation, parameterised by configuration and shared across every RL-trained baseline (confound 6). Heuristic baselines do not consume the reward at decision time but their episodes are scored with it so that metrics remain comparable.
- `C_time` accrues every step regardless of action, including `WAIT`, so indefinite waiting carries a nonzero opportunity cost. This is a binding requirement — waiting forever must not be optimal — and is verified directly by a unit test asserting that an all-`WAIT` episode's return is strictly less than a successful episode's.
- Shaping beyond these five terms is deliberately avoided. Any shaping added later must be a named, separately ablatable term rather than folded into an existing coefficient.

> **Divergence.** `C_collision` never fires on the fast backend, which has no collision model. The term is wired through the whole pipeline and is ready, but is inert in every result reported. `EpisodeConfig` deliberately declares no collision margin, on the grounds that a declared margin would advertise a check nothing performs.

## 7. Episode structure

An episode terminates on exactly one of `success`, `target_lost`, `collision`, `timeout`, or `irrecoverable_state`. The reasons are checked in a fixed priority order so that exactly one is ever reported even when several conditions hold simultaneously, and the reason is recorded on every episode; episodes never terminate with an unclassified reason. Defaults: `max_duration_s = 60.0`, `target_lost_radius_m = 3.0`.

## 8. Baseline ladder

Implemented in order, each validated before the next was attempted. All share the action space, the observation schema (minus the opportunity block where applicable), the reward, and — for RL baselines — the training budget and disturbance distribution.

| ID | Name | Decision rule | Learned | Built |
|---|---|---|---|---|
| B0 | Immediate heuristic | `ATTEMPT_GRASP` iff `good_opportunity`, else `WAIT` | No | Yes |
| B1 | Reactive heuristic | Configurable thresholds on distance and speed; reduces to B0 at default thresholds | No | Yes |
| B2 | Standard RL (reactive) | PPO over the base observation only | Yes | Yes |
| B3 | Temporal RL | B2 plus observation history (stacked frames or a recurrent policy) | Yes | **No** |
| B4 | Opportunity-aware RL | B2 plus the appended estimator output | Yes | Yes |
| — | Oracle timing reference | Commits exactly when a grasp committed now would succeed, found by playing the commitment forward on a copy of the deterministic env | Not a policy | Yes (`algorithms/oracle.py`) |

B3 was conditional on evidence of meaningful temporal dependence; that evidence was not gathered, and B3 was not built.

The oracle is the reference against which timing regret (H4) is measured. It reads the true future, which is permitted only because it is an evaluation reference: no policy observes it or trains against it. Two limits bound it. It optimises timing only (`WAIT` / `ATTEMPT_GRASP`, never `APPROACH`), which is tight in the shipped configurations, where the standby pose is already well inside the workspace radius. And it is exact only on a deterministic backend such as `FastSimulator`.

## 9. Divergences from specification, collected

1. **Feasibility predicate reduced** from a four-condition geometric definition to distance plus relative speed; `graspable`, orientation error, and angular velocity are not evaluated (Section 5).
2. **End-effector velocity is a zero proxy**, so relative velocity is effectively target velocity in the vessel frame (Section 5).
3. **Collision penalty inert** — the fast backend has no contact model, so `C_collision` never fires (Section 6).
4. **B3 was not built** (Section 8). The oracle was added later, so H4 is now testable but has not yet been run at scale.
5. **Uncertainty observation fields never enabled** (Section 3).
6. **Only one of two backends exists** — no policy has been trained or evaluated through ROS2/Gazebo (see [architecture.md §2](architecture.md#2-the-two-backend-design)).

## 10. Training and evaluation separation

A training run and the evaluation used to report its performance never share a seed or a disturbance draw. Training draws randomised disturbances across seeds; evaluation uses fixed held-out seeds offset by `+10_000`. Training-time episode statistics may be logged for monitoring but are never reported as a result.

## 11. Curriculum

Staged difficulty was specified as configurable rather than hard-coded, progressing from static through low target motion, low vessel disturbance, combined motion, higher disturbance, observation noise, and distribution shift.

> **Divergence.** No staged curriculum was implemented. What exists is the three-rung severity ladder (`moving_target_mild`, `moving_target`, `moving_target_severe`) used as separate experimental conditions rather than as a progression, plus `sydney_regatta_default`. The later stages — observation noise and distribution shift — were never built, which is why H3 is untested.

## 12. Interfaces

```python
class SimulationInterface(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def step(self, dt: float) -> None: ...
    def get_robot_state(self) -> RobotState: ...
    def get_target_state(self) -> TargetState: ...
    def get_environment_state(self) -> EnvironmentState: ...
    def apply_action(self, action: int) -> None: ...


class DynamicGraspEnv(gymnasium.Env):
    def reset(self, *, seed: int | None = None, options: dict | None = None): ...
    def step(self, action: int): ...

    observation_space: gymnasium.spaces.Box
    action_space: gymnasium.spaces.Discrete


class RLAlgorithm(Protocol):
    def train(self, total_steps: int) -> None: ...
    def evaluate(self, n_episodes: int, seed: int) -> EvaluationResult: ...
    def save_checkpoint(self, path: Path) -> None: ...
    def load_checkpoint(self, path: Path) -> None: ...
```

Both `SimulationInterface` and `RLAlgorithm` are real `typing.Protocol` definitions rather than documentation-only contracts, so conformance is checked structurally by mypy. All `RLAlgorithm` configuration surfaces — seed, environment config, algorithm config, network config, training steps, evaluation and checkpoint frequency — are config-driven rather than hard-coded.
