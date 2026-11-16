"""The oracle timing reference from docs/rl-formulation.md Section 8: an
upper bound on how well grasp *timing* can be done, used to measure
timing regret (H4). It is an evaluation reference, never a policy input
and never something a learned policy is trained against. That is the
only place in the codebase allowed to read the true future
(rl-formulation.md Section 4, leakage rule; confound #7).

How it sees the future. The fast backend is deterministic once an
episode has been reset with a seed: nothing random happens after
`reset`. So "what would happen if I committed to a grasp right now?" has
an exact answer, which the oracle gets by deep-copying the whole
environment and playing the commitment forward on the copy. The real
environment is untouched by the lookahead.

Decision rule. At every decision step, commit (ATTEMPT_GRASP) if and
only if a grasp committed now would succeed; otherwise WAIT. Under the
reward in env/reward.py this is the best achievable timing:
- a successful grasp is worth +1 and can happen at most once;
- every step costs the same time penalty whatever the action;
- a failed grasp only subtracts, and WAIT leaves the arm still, so it
  adds no motion cost.
So the earliest step at which a commitment succeeds gives the highest
return reachable by WAIT/ATTEMPT_GRASP timing alone. The oracle never
makes a failed attempt.

Scope, stated so the bound is not over-read:
- It optimises *timing* only. It never uses APPROACH, so on a task where
  repositioning the arm first could open an earlier window, a policy
  could in principle beat it. In the shipped configurations the standby
  pose is already well inside the workspace radius, so APPROACH buys
  nothing and the bound is tight in practice.
- It assumes a deterministic backend. It is exact on FastSimulator and
  would not be on a noisy backend such as ROS2/Gazebo.
"""

from __future__ import annotations

import copy

from opportunity_rl.algorithms.heuristic import _score_episodes
from opportunity_rl.algorithms.interface import EvaluationResult
from opportunity_rl.sim.interface import ATTEMPT_GRASP, WAIT

# Upper bound on how many decision steps a committed grasp can take before
# it resolves. Far above any grasp_duration_steps in use; it only guards
# against an endless lookahead if a backend never resolves a commitment.
_MAX_LOOKAHEAD_STEPS = 1_000


def grasp_would_succeed(env) -> bool:
    """Would ATTEMPT_GRASP issued in `env`'s current state succeed?

    Answered exactly by playing the commitment forward on a deep copy:
    ATTEMPT_GRASP, then WAIT (ignored while the maneuver is committed)
    until the episode reports success, a failed grasp, or any other
    ending. `env` itself is not modified."""
    lookahead = copy.deepcopy(env)
    action = ATTEMPT_GRASP
    for _ in range(_MAX_LOOKAHEAD_STEPS):
        _, _, terminated, truncated, info = lookahead.step(action)
        if info.get("termination_reason") == "success":
            return True
        if info.get("reward_failure", 0.0) > 0.0:
            return False
        if terminated or truncated:
            return False  # e.g. timed out mid-maneuver
        if not _in_maneuver(lookahead):
            # The commitment resolved without success or failure. That
            # cannot happen on FastSimulator, but don't loop forever on a
            # backend where it can.
            return False
        action = WAIT
    return False


def _in_maneuver(env) -> bool:
    """True while a committed multi-step grasp is still in progress."""
    return getattr(env._sim, "_grasp_steps_remaining", 0) > 0


class OracleTimingPolicy:
    """Commit exactly when a commitment would succeed, else WAIT. Same
    `evaluate` shape and metrics as B0/B1, so it slots into the
    evaluation matrix next to them. See module docstring."""

    def __init__(self, env) -> None:
        self._env = env

    def _decide(self, _feasibility) -> int:
        return ATTEMPT_GRASP if grasp_would_succeed(self._env) else WAIT

    def evaluate(self, n_episodes: int, seed: int) -> EvaluationResult:
        return _score_episodes(self._env, self._decide, n_episodes, seed)


def timing_regret(oracle: EvaluationResult, policy: EvaluationResult) -> float:
    """Mean return the policy left on the table relative to the oracle.

    Only meaningful when both were evaluated on the SAME episode seeds
    (the same disturbance draws); then the difference of means equals the
    mean of the per-episode differences. Non-negative up to the oracle's
    timing-only scope (see module docstring)."""
    if oracle.n_episodes != policy.n_episodes:
        raise ValueError(
            f"oracle ran {oracle.n_episodes} episodes, policy {policy.n_episodes}; "
            "timing regret needs the same episode seeds for both"
        )
    return oracle.mean_return - policy.mean_return


def gap_closed(reactive_return: float, aware_return: float, oracle_return: float) -> float:
    """H4: the fraction of the reactive policy's gap to the oracle that
    the opportunity-aware policy closes.

        (aware - reactive) / (oracle - reactive)

    1.0 means the aware policy matches the oracle, 0.0 means it is no
    better than the reactive policy, negative means it is worse. Raises if
    the reactive policy already matches the oracle, since there is then no
    gap to close and the ratio is undefined."""
    gap = oracle_return - reactive_return
    if abs(gap) < 1e-12:
        raise ValueError("reactive policy already matches the oracle: no gap to close")
    return (aware_return - reactive_return) / gap
