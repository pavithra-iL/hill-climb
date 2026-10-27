"""Command-line interface for running the hill climbing optimizer.

Examples:
    python cli.py --objective bumpy --restarts 25
    python cli.py --objective gaussian --start 3 --step-size 0.01
"""

from __future__ import annotations

import argparse

from hill_climb import hill_climb, random_restart_hill_climb
from objectives import OBJECTIVES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a hill climbing optimizer.")
    parser.add_argument(
        "--objective",
        choices=sorted(OBJECTIVES),
        default="bumpy",
        help="Which objective function to maximize.",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Starting x value.")
    parser.add_argument(
        "--step-size", type=float, default=0.01, help="Neighbor step size."
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Iteration cap per run."
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=0,
        help="If > 0, use random-restart search with this many restarts.",
    )
    parser.add_argument(
        "--bounds",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(-5.0, 5.0),
        help="Search bounds for random restarts.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    objective = OBJECTIVES[args.objective]

    if args.restarts > 0:
        result = random_restart_hill_climb(
            objective,
            bounds=tuple(args.bounds),
            restarts=args.restarts,
            step_size=args.step_size,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )
        mode = f"random restart x{args.restarts}"
    else:
        result = hill_climb(
            objective,
            start=args.start,
            step_size=args.step_size,
            max_iterations=args.max_iterations,
        )
        mode = f"basic from {args.start}"

    print(f"objective={args.objective} ({mode})")
    print(f"  x         = {result.x:.6f}")
    print(f"  value     = {result.value:.6f}")
    print(f"  iterations= {result.iterations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
