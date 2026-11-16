#!/usr/bin/env python3
"""H4: timing regret against the oracle, and how much of the reactive
policy's gap to the oracle the opportunity-aware policy closes.

For each seed this trains B2 (reactive PPO) and B4 (opportunity-aware
PPO) exactly as train_h1_comparison.py does, then evaluates B2, B4, the
B0 heuristic and the oracle timing reference on the SAME held-out
episode seeds (`seed + 10_000`), so every return is measured on the
same disturbance draws and regrets are directly comparable.

    python3 experiments/oracle_regret.py --disturbance moving_target \\
        --grasp-duration-steps 3 --total-steps 80000 --seeds 0 1 2

Reported per seed and averaged:
- timing regret = oracle return - policy return (lower is better)
- gap closed    = (B4 - B2) / (oracle - B2)   (research-question.md H4)

The oracle reads the true future and is an evaluation reference only;
neither policy ever sees it (see src/opportunity_rl/algorithms/oracle.py).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from opportunity_rl.algorithms.heuristic import B0ImmediatePolicy
from opportunity_rl.algorithms.oracle import OracleTimingPolicy, gap_closed, timing_regret
from opportunity_rl.algorithms.ppo import PPO, PPOConfig
from opportunity_rl.env.env import DynamicGraspEnv, EnvConfig
from opportunity_rl.opportunity.estimator import OpportunityConfig, OpportunityEstimator
from opportunity_rl.provenance import git_commit, git_is_dirty
from opportunity_rl.sim.disturbance_config import load_disturbance_config
from opportunity_rl.sim.fast_simulator import FastSimulator

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EVAL_SEED_OFFSET = 10_000


def _make_env(disturbance: str, grasp_steps: int, opportunity: bool) -> DynamicGraspEnv:
    # Same construction as train_h1_comparison.py / learning_curve_h1.py:
    # B2 and B4 differ only by the appended opportunity block (confound #1).
    sim = FastSimulator(load_disturbance_config(disturbance), grasp_duration_steps=grasp_steps)
    if opportunity:
        opp = OpportunityConfig()
        return DynamicGraspEnv(
            sim,
            config=EnvConfig(),
            opportunity_estimator=OpportunityEstimator(opp),
            opportunity_dim=opp.dim,
        )
    return DynamicGraspEnv(sim, config=EnvConfig())


def _gap_closed_or_none(b2: float, b4: float, oracle: float) -> float | None:
    try:
        return gap_closed(b2, b4, oracle)
    except ValueError:
        return None  # B2 already matches the oracle: nothing to close


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--disturbance", type=str, default="moving_target")
    parser.add_argument("--grasp-duration-steps", type=int, default=3)
    parser.add_argument("--total-steps", type=int, default=80_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    tag = f"{args.disturbance}_g{args.grasp_duration_steps}_{args.total_steps // 1000}k"
    run_name = args.run_name or f"oracle_regret_{tag}_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    per_seed: dict[int, dict[str, float | None]] = {}
    t0 = time.time()
    for seed in args.seeds:
        eval_seed = seed + EVAL_SEED_OFFSET
        results = {}
        for name, aware in (("B2", False), ("B4", True)):
            ppo = PPO(
                _make_env(args.disturbance, args.grasp_duration_steps, aware),
                config=PPOConfig(seed=seed),
            )
            ppo.train(args.total_steps)
            results[name] = ppo.evaluate(args.eval_episodes, seed=eval_seed)
        results["B0"] = B0ImmediatePolicy(
            _make_env(args.disturbance, args.grasp_duration_steps, False)
        ).evaluate(args.eval_episodes, seed=eval_seed)
        results["oracle"] = OracleTimingPolicy(
            _make_env(args.disturbance, args.grasp_duration_steps, False)
        ).evaluate(args.eval_episodes, seed=eval_seed)

        returns = {name: r.mean_return for name, r in results.items()}
        row: dict[str, float | None] = dict(returns)
        for name in ("B0", "B2", "B4"):
            row[f"regret_{name}"] = timing_regret(results["oracle"], results[name])
        row["gap_closed"] = _gap_closed_or_none(returns["B2"], returns["B4"], returns["oracle"])
        per_seed[seed] = row

        gc = row["gap_closed"]
        print(
            f"[{tag} seed{seed}] oracle={returns['oracle']:+.3f} "
            f"B0={returns['B0']:+.3f} B2={returns['B2']:+.3f} B4={returns['B4']:+.3f} | "
            f"regret B2={row['regret_B2']:+.3f} B4={row['regret_B4']:+.3f} | "
            f"gap closed={'n/a' if gc is None else f'{gc:+.2f}'}"
        )

    def _mean(key: str) -> float | None:
        vals = [v[key] for v in per_seed.values() if v[key] is not None]
        return statistics.mean(vals) if vals else None  # type: ignore[arg-type]

    summary = {
        key: _mean(key)
        for key in ("oracle", "B0", "B2", "B4", "regret_B0", "regret_B2", "regret_B4")
    }
    # Gap closed from the seed-averaged returns: a ratio of means is more
    # stable than a mean of per-seed ratios, whose denominators can be tiny.
    summary["gap_closed"] = (
        _gap_closed_or_none(summary["B2"], summary["B4"], summary["oracle"])  # type: ignore[arg-type]
        if summary["oracle"] is not None
        else None
    )

    print(f"\nMean over seeds {args.seeds}:")
    for key, value in summary.items():
        print(f"  {key:<10} {'n/a' if value is None else f'{value:+.3f}'}")

    metadata = {
        "run_name": run_name,
        "git_commit": git_commit(REPO_ROOT),
        "git_dirty": git_is_dirty(REPO_ROOT),
        "disturbance": args.disturbance,
        "grasp_duration_steps": args.grasp_duration_steps,
        "total_steps": args.total_steps,
        "seeds": args.seeds,
        "eval_episodes": args.eval_episodes,
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "summary": summary,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nwrote {run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
