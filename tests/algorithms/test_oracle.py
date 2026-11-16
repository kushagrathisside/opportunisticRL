"""Oracle timing reference tests. Require the `rl` extra (gymnasium, for
the env)."""

import copy

import pytest

pytest.importorskip("gymnasium")

from opportunity_rl.algorithms.heuristic import B0ImmediatePolicy  # noqa: E402
from opportunity_rl.algorithms.interface import EvaluationResult  # noqa: E402
from opportunity_rl.algorithms.oracle import (  # noqa: E402
    OracleTimingPolicy,
    gap_closed,
    grasp_would_succeed,
    timing_regret,
)
from opportunity_rl.env.env import DynamicGraspEnv, EnvConfig  # noqa: E402
from opportunity_rl.sim.disturbance_config import load_disturbance_config  # noqa: E402
from opportunity_rl.sim.fast_simulator import ATTEMPT_GRASP, WAIT, FastSimulator  # noqa: E402


def _env(disturbance="moving_target", grasp_duration_steps=3):
    sim = FastSimulator(
        load_disturbance_config(disturbance), grasp_duration_steps=grasp_duration_steps
    )
    return DynamicGraspEnv(sim, config=EnvConfig())


def _play_commit(env):
    """Commit for real in `env`; True if the grasp succeeded."""
    action = ATTEMPT_GRASP
    while True:
        _, _, terminated, truncated, info = env.step(action)
        if info["termination_reason"] == "success":
            return True
        if info["reward_failure"] > 0 or terminated or truncated:
            return False
        action = WAIT


def test_lookahead_does_not_modify_the_real_env():
    env = _env()
    env.reset(seed=3)
    t_before = env._sim.get_environment_state().sim_time
    obs_before = env._build_observation()
    grasp_would_succeed(env)
    assert env._sim.get_environment_state().sim_time == t_before
    assert (env._build_observation() == obs_before).all()


@pytest.mark.parametrize("grasp_duration_steps", [1, 3])
def test_prediction_matches_what_actually_happens(grasp_duration_steps):
    # The oracle's forecast must agree with reality at every state it is
    # asked about, both when it says "yes" and when it says "no".
    checked_yes = checked_no = 0
    for seed in range(15):
        env = _env(grasp_duration_steps=grasp_duration_steps)
        env.reset(seed=seed)
        for _ in range(40):
            predicted = grasp_would_succeed(env)
            actual = _play_commit(copy.deepcopy(env))
            assert predicted == actual
            checked_yes += predicted
            checked_no += not predicted
            _, _, terminated, truncated, _ = env.step(WAIT)
            if terminated or truncated:
                break
    assert checked_yes > 0 and checked_no > 0  # both branches exercised


def test_oracle_never_fails_a_grasp_and_always_succeeds():
    result = OracleTimingPolicy(_env()).evaluate(n_episodes=20, seed=10_000)
    assert result.success_rate == 1.0
    assert result.termination_reason_counts == {"success": 20}


@pytest.mark.parametrize("disturbance", ["moving_target_mild", "moving_target_severe"])
def test_oracle_is_at_least_as_good_as_b0(disturbance):
    # B0 is reactive and can commit into a window that closes mid-grasp;
    # the oracle cannot. On the same seeds it must never do worse.
    seed = 10_000
    oracle = OracleTimingPolicy(_env(disturbance)).evaluate(n_episodes=30, seed=seed)
    b0 = B0ImmediatePolicy(_env(disturbance)).evaluate(n_episodes=30, seed=seed)
    assert oracle.mean_return >= b0.mean_return - 1e-9
    assert timing_regret(oracle, b0) >= -1e-9


def _result(mean_return, n=10):
    return EvaluationResult(
        n_episodes=n, success_rate=1.0, mean_return=mean_return, mean_episode_length=10.0
    )


def test_timing_regret_is_oracle_minus_policy():
    assert timing_regret(_result(0.9), _result(0.6)) == pytest.approx(0.3)


def test_timing_regret_refuses_mismatched_episode_counts():
    with pytest.raises(ValueError):
        timing_regret(_result(0.9, n=10), _result(0.6, n=20))


def test_gap_closed():
    assert gap_closed(reactive_return=0.5, aware_return=0.5, oracle_return=0.9) == 0.0
    assert gap_closed(reactive_return=0.5, aware_return=0.9, oracle_return=0.9) == 1.0
    assert gap_closed(reactive_return=0.5, aware_return=0.7, oracle_return=0.9) == pytest.approx(
        0.5
    )
    assert gap_closed(reactive_return=0.5, aware_return=0.4, oracle_return=0.9) < 0.0
    with pytest.raises(ValueError):
        gap_closed(reactive_return=0.9, aware_return=0.9, oracle_return=0.9)
