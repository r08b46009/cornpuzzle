#!/usr/bin/env python3

import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--puzzles", required=True)
parser.add_argument("--answers", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()


puzzles = sorted(
    Path(args.puzzles).rglob("puzzle_*.txt")
)


out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)


count = 0

with out.open("w") as f:
    f.write("# task_id\tpuzzle\tsolution\tprefix\n")

    for p in puzzles:

        sol = Path(args.answers) / (
            p.stem + "_solution.txt"
        )

        if not sol.exists():
            print(
                f"[skip] missing solution {sol}"
            )
            continue

        f.write(
            f"{p.stem}\t"
            f"{p}\t"
            f"{sol}\t"
            f"0\n"
        )

        count += 1


print(
    f"[full eval] wrote {count} tasks"
)
