#!/usr/bin/env python3
"""M7: the evaluation matrix -- run every available baseline against a
disturbance config across multiple seeds and print a comparison table.

Baselines evaluated: random floor, B0 (immediate heuristic), B1 (reactive
threshold heuristic), the oracle timing reference (an upper bound that
reads the true future; see algorithms/oracle.py), and B2 (PPO, if a
trained checkpoint is provided).
This is the shared harness every later milestone's comparison reuses --
M9's H1 comparison (B2 vs B4) is the same matrix with B4 added.

All baselines are scored with the identical EvaluationResult metrics on
the identical held-out eval seeds, so the table compares like with like
(confound register: shared reward #6, shared decision rate #2 -- both
enforced by the shared env, not per-baseline).

The heuristic baselines (random/B0/B1) need no training, so this runs in
seconds. B2 requires a checkpoint from experiments/train_b2.py; if none
is given, its row is skipped (the matrix still reports the heuristics).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from opportunity_rl.algorithms.heuristic import B0ImmediatePolicy, B1ThresholdPolicy
from opportunity_rl.algorithms.interface import EvaluationResult
from opportunity_rl.algorithms.oracle import OracleTimingPolicy
from opportunity_rl.algorithms.random_policy import RandomPolicy
from opportunity_rl.env.env import DynamicGraspEnv, EnvConfig
from opportunity_rl.sim.disturbance_config import load_disturbance_config
from opportunity_rl.sim.fast_simulator import FastSimulator

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EVAL_SEED_OFFSET = 10_000


def _make_env(disturbance: str, grasp_duration_steps: int) -> DynamicGraspEnv:
    return DynamicGraspEnv(
        FastSimulator(
            load_disturbance_config(disturbance), grasp_duration_steps=grasp_duration_steps
        ),
        config=EnvConfig(),
    )


def _aggregate(results: list[EvaluationResult]) -> dict[str, float]:
    """Aggregate one baseline's per-seed results into mean +/- std."""
    returns = [r.mean_return for r in results]
    successes = [r.success_rate for r in results]
    lengths = [r.mean_episode_length for r in results]
    return {
        "mean_return": statistics.mean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "mean_success_rate": statistics.mean(successes),
        "mean_episode_length": statistics.mean(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--disturbance", type=str, default="moving_target")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument(
        "--grasp-duration-steps",
        type=int,
        default=1,
        help=(
            "decision steps a committed ATTEMPT_GRASP takes. MUST match the "
            "value the policies being compared were trained/evaluated at -- "
            "the H1 comparison uses 3, and a matrix run at the default 1 is "
            "measuring a different task and is not comparable to it."
        ),
    )
    parser.add_argument(
        "--b2-checkpoint",
        type=str,
        default=None,
        help="path to a train_b2.py checkpoint; if omitted, B2 row is skipped",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    baselines: dict[str, list[EvaluationResult]] = {
        "random": [],
        "B0": [],
        "B1": [],
        "oracle": [],
    }

    def make_env() -> DynamicGraspEnv:
        return _make_env(args.disturbance, args.grasp_duration_steps)

    for seed in args.seeds:
        eval_seed = seed + EVAL_SEED_OFFSET
        baselines["random"].append(
            RandomPolicy(make_env(), seed=seed).evaluate(args.eval_episodes, seed=eval_seed)
        )
        baselines["B0"].append(
            B0ImmediatePolicy(make_env()).evaluate(args.eval_episodes, seed=eval_seed)
        )
        baselines["B1"].append(
            B1ThresholdPolicy(make_env()).evaluate(args.eval_episodes, seed=eval_seed)
        )
        baselines["oracle"].append(
            OracleTimingPolicy(make_env()).evaluate(args.eval_episodes, seed=eval_seed)
        )

    if args.b2_checkpoint:
        from opportunity_rl.algorithms.ppo import PPO, PPOConfig

        baselines["B2"] = []
        for seed in args.seeds:
            env = _make_env(args.disturbance, args.grasp_duration_steps)
            ppo = PPO(env, config=PPOConfig(seed=seed))
            ppo.load_checkpoint(Path(args.b2_checkpoint))
            baselines["B2"].append(ppo.evaluate(args.eval_episodes, seed=seed + EVAL_SEED_OFFSET))

    aggregated = {name: _aggregate(results) for name, results in baselines.items()}

    print(
        f"\nEvaluation matrix -- disturbance={args.disturbance}, "
        f"grasp_duration_steps={args.grasp_duration_steps}, "
        f"seeds={args.seeds}, {args.eval_episodes} eval episodes each\n"
    )
    header = f"{'baseline':<10} {'return (mean±std)':>22} {'success':>9} {'ep_len':>8}"
    print(header)
    print("-" * len(header))
    for name, agg in aggregated.items():
        print(
            f"{name:<10} "
            f"{agg['mean_return']:>10.2f} ± {agg['std_return']:<8.2f} "
            f"{agg['mean_success_rate']:>9.3f} {agg['mean_episode_length']:>8.1f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "disturbance": args.disturbance,
                    "grasp_duration_steps": args.grasp_duration_steps,
                    "seeds": args.seeds,
                    "eval_episodes": args.eval_episodes,
                    "aggregated": aggregated,
                },
                indent=2,
            )
        )
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
