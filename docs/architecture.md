# System Architecture

This document describes the layered architecture, the boundaries between languages and frameworks, and the confound register that governs whether a comparison between baselines may be reported as research evidence. See [research-question.md](research-question.md) for the question the architecture exists to answer and [project-plan.md](project-plan.md) for the order in which it was built.

## 1. Layered view

The system is organised into four layers, each depending only on the layer directly below it through a narrow, named interface.

```text
┌─────────────────────────────────────────────────────────┐
│  RESEARCH / RL LAYER                                     │
│  baseline ladder · opportunity estimator · training      │
│  src/opportunity_rl/{algorithms,opportunity}             │
└───────────────────────────┬─────────────────────────────┘
                            │  DynamicGraspEnv (Gymnasium API)
┌───────────────────────────▼─────────────────────────────┐
│  TASK / ENVIRONMENT LAYER                                │
│  observation schema · reward · episode termination       │
│  src/opportunity_rl/env                                  │
└───────────────────────────┬─────────────────────────────┘
                            │  SimulationInterface (Protocol)
┌───────────────────────────▼─────────────────────────────┐
│  ROBOTICS / ROS2 LAYER                                   │
│  state estimation · TF2 · MoveIt2 grasp pipeline         │
│  ros2_ws/src/{marine_state_estimator,marine_manipulation,│
│  wamv_manipulator_description,wamv_manipulator_moveit_   │
│  config}                                                 │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│  SIMULATION / PHYSICS LAYER                              │
│  Gazebo Harmonic · VRX marine environment · WAM-V        │
│  fetched via ros2_ws/vrx.repos (not vendored)            │
└─────────────────────────────────────────────────────────┘
```

The RL layer never imports ROS2 or simulator-specific types; it depends only on `DynamicGraspEnv`, a plain [Gymnasium](https://gymnasium.farama.org/) environment. This is the single most important structural decision in the codebase: **training must be able to run without ROS2 or Gazebo in the loop.** ROS2 and Gazebo constitute one *implementation* of `SimulationInterface`, intended for integration testing and evaluation under the real state-estimation and control stack, not for every training step.

## 2. The two-backend design

```text
                      ┌──────────────────────┐
                      │   DynamicGraspEnv     │
                      │   (Gymnasium API)     │
                      └───────────┬──────────┘
                                  │  SimulationInterface
                  ┌───────────────┴───────────────┐
        ┌─────────▼─────────┐          ┌──────────▼──────────┐
        │  Fast kinematic    │          │  ROS2 + Gazebo       │
        │  backend           │          │  execution           │
        │  (training path,   │          │  (integration and    │
        │  pure Python)      │          │  evaluation path)    │
        │  IMPLEMENTED       │          │  NOT IMPLEMENTED     │
        └────────────────────┘          └──────────────────────┘
```

`SimulationInterface` is a real `typing.Protocol` (`src/opportunity_rl/sim/interface.py`) with methods `reset`, `step`, `get_robot_state`, `get_target_state`, `get_environment_state` and `apply_action`, so mypy verifies conformance structurally rather than by assertion. `DynamicGraspEnv` is identical regardless of which backend is supplied.

**Only the fast backend exists.** The ROS2/Gazebo backend satisfying the same interface was deliberately deferred and never built; the ROS2 layer was validated independently for state estimation and for a scripted grasp against a static target, but no policy has been trained or evaluated through it. This is the principal architectural limitation of the work and is recorded as a threat to validity in [evaluation.md §9](evaluation.md#9-threats-to-validity-and-conditions-for-citing-these-results).

Reward and opportunity computation live in the task layer, never inside a `SimulationInterface` implementation. The simulator tracks the physical fact of whether the gripper holds the target; whether that constitutes *success* is a task-layer decision. The separation is verified by construction — the simulator imports no reward code.

## 3. Language and framework boundaries

Python is used throughout the RL and task layers, and for the ROS2 nodes that do not require hard real-time performance. No C++ exists in this repository, and none is planned until a concrete need arises, such as a controller loop requiring tighter timing than `rclpy` provides.

MoveIt2 and existing ROS2 controller packages provide trajectory generation, inverse kinematics, and low-level control. No novel low-level grasp controller, IK solver, or physics engine is implemented here. The RL policy's action space terminates at the high-level decision — `WAIT`, `APPROACH`, `ATTEMPT_GRASP`, `ABORT` — and hands off to `marine_manipulation`, which is conventional robotics code.

## 4. Confound register

A comparison between baselines is admissible as evidence only if every item below is either resolved, with a stated mechanism, or explicitly accepted and documented. The register is the operational form of the requirement in [research-question.md §4](research-question.md#4-criteria-for-support-and-rejection) that a comparison contaminated by an uncorrected confound be reported as inconclusive rather than folded into a verdict.

Each item is resolved *structurally* wherever possible — that is, by making the confound impossible to reintroduce (one shared implementation rather than one per baseline), not by policing it in review.

| # | Confound | Status | Mechanism |
|---|---|---|---|
| 1 | Opportunity-aware policy receives more *unrelated* state than the reactive policy | **Resolved** | Both policies read the same `BASE_FIELDS` schema; the opportunity signal is appended as a separate block (`append_opportunity`), never interleaved. The two envs are built by one code path with and without the estimator, and a unit test asserts the base blocks are byte-identical across steps. |
| 2 | Different action frequencies between policies | **Resolved** | A single `decision_period_s` configuration value; the env advances the simulator by exactly that amount per env-step, so every baseline is stepped at the same decision rate by construction. |
| 3 | Different controller capabilities between methods | **Resolved on the fast backend** | All baselines act through the same shared manipulator proxy and the same feasibility predicate. On the ROS2 backend all methods would hand off to the same `marine_manipulation` stack, but no baseline has been run there. |
| 4 | Different training disturbance distributions | **Resolved** | Named configurations in `configs/disturbances/*.yaml`, referenced by name from each experiment rather than redefined per experiment. An unknown name fails loudly rather than falling back to defaults, so a run's configuration claim stays trustworthy. |
| 5 | Different compute budgets per algorithm | **Resolved** | Total environment steps, not wall-clock time, are matched across compared runs and recorded in each run's metadata. |
| 6 | Reward differs between baselines | **Resolved** | One shared `reward.compute_reward`, parameterised by config, returning each term separately for failure analysis. Heuristic baselines do not consume the reward at decision time but their episodes are scored with it, so metrics remain comparable. |
| 7 | Opportunity estimator leaks future ground truth into the online policy | **Resolved** | The estimator is called with current state only; the env holds no future state, so leakage is structurally impossible rather than merely prohibited. The estimator reads current and one-step-past velocity and extrapolates. The oracle reference (`algorithms/oracle.py`) is the only component permitted to read true future state. It does so by deep-copying the env for lookahead, is used only to score episodes, and is never an input to a policy. |
| 8 | Simulator state exposed to one method but not another | **Resolved** | All policies read observations exclusively through the `Observation` schema produced by `DynamicGraspEnv`; no baseline receives direct backend access. |

Two qualifications apply to the register as a whole. First, resolution of items 1–8 establishes that the *comparison* is sound; it says nothing about whether the *measurements* are trustworthy, and they are currently not, for the provenance reasons in [evaluation.md §2](evaluation.md#2-provenance-and-status-of-the-evidence). Second, item 3 is resolved only on the backend that was actually used.

## 5. Simulator selection

The requirement was a ROS2-compatible simulator with a maritime environment. The selection was made against the environment as it actually exists rather than against documentation.

1. **ROS2 Jazzy Jalisco** was already installed, which targets Ubuntu 24.04 and pairs with **Gazebo Harmonic** (`gz-sim`) rather than the end-of-life Gazebo Classic.
2. **Gazebo Harmonic 8.11.0** was installed and verified (`gz sim --version`). It arrived through ROS Jazzy's vendor packages (`ros-jazzy-gz-sim-vendor` and related), which build Gazebo inside the ROS install prefix rather than through the separate OSRF apt repository. The `gz` binary is not on `PATH` unless `/opt/ros/jazzy/setup.bash` is sourced.
3. **VRX (Virtual RobotX)** was selected for the maritime environment: it is the standard ROS2 and Gazebo maritime stack and supplies a USV platform (WAM-V) together with wave and current disturbance models. It has no apt package and was built from source at the `jazzy` branch pinned in `ros2_ws/vrx.repos`.
4. **VRX ships no manipulator.** A Franka Emika Panda was mounted on the WAM-V deck via a single fixed joint, using the description package MoveIt2 maintains for its own tutorials and CI. The rationale and the rejected alternative are recorded in [implementation-status.md](implementation-status.md#2-manipulator-selection).

A paper-by-paper survey of how marine-robotics simulators model ocean waves, including a source-verified account of what this project's VRX installation actually implements (Pierson-Moskowitz, not JONSWAP as some secondary sources state), is in [wave-simulation-literature-review.md](wave-simulation-literature-review.md). It directly informed the disturbance-severity parameters.

## 6. Coordinate and unit conventions

The full TF2 frame tree, coordinate conventions, units, and the `marine_state_estimator` topic contract are documented in [ros2.md](ros2.md). One finding affects every layer above the ROS2 one and is therefore surfaced here: **this simulation contains no world-fixed TF2 frame.** `wamv/wamv/base_link` is the root of the TF tree, and the vessel's world-frame pose is obtainable only through a Gazebo Transport workaround, not through standard ROS2 TF2.

The practical consequence is that the vessel-relative frame, not any world frame, is the load-bearing reference frame for this project. This matches the `ee_pose` observation field's specification in vessel-base coordinates, which was fixed before the frame situation was established and proved to be the correct choice.
