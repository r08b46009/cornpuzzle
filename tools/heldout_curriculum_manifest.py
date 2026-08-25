#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


STAGES = [
    ("remain2",),
    ("remain2", "remain4"),
    ("remain4", "remain6"),
    ("remain6", "remain8"),
    ("remain8", "remain10"),
    ("remain10", "remain12"),
    ("remain12", "remain14"),
    ("remain14", "remain16"),
    ("remain16", "remain18"),
    ("remain18", "remain20"),
    ("remain20", "remain22"),
    ("remain22", "full"),
    ("full",),
]


def find_solution(answers: Path, stem: str):
    expected = f"{stem}_solution.txt"

    direct = answers / expected
    if direct.exists():
        return direct.resolve()

    hits = list(answers.rglob(expected))

    if len(hits) == 1:
        return hits[0].resolve()

    if not hits:
        return None

    raise RuntimeError(
        f"multiple solutions found for {stem}: {hits}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-bank", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--puzzles", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected", type=int, default=20)
    args = p.parse_args()

    tasks = json.loads(Path(args.task_bank).read_text())
    state = json.loads(Path(args.state).read_text())

    active_levels = STAGES[int(state["stage_index"])]

    # Derive the proper CPREFIX for each level from the
    # existing training task bank.
    prefix_by_level = {}

    for t in tasks:
        level = t["level"]
        prefix = int(t["prefix"])

        if level in prefix_by_level:
            if prefix_by_level[level] != prefix:
                raise RuntimeError(
                    f"inconsistent prefix for {level}"
                )
        else:
            prefix_by_level[level] = prefix

    puzzle_dir = Path(args.puzzles)
    answer_dir = Path(args.answers)

    puzzles = sorted(
        p for p in puzzle_dir.iterdir()
        if p.is_file()
        and p.suffix == ".txt"
        and "_solution" not in p.stem
    )

    if len(puzzles) != args.expected:
        raise RuntimeError(
            f"expected {args.expected} held-out puzzles in "
            f"{puzzle_dir}, got {len(puzzles)}"
        )

    rows = []

    for level in active_levels:
        if level not in prefix_by_level:
            raise RuntimeError(
                f"level {level} missing from task bank"
            )

        prefix = prefix_by_level[level]

        for puzzle in puzzles:
            # For full, solution is unused because prefix=0.
            if level == "full":
                solution = puzzle.resolve()
            else:
                solution = find_solution(
                    answer_dir,
                    puzzle.stem,
                )

                if solution is None:
                    raise RuntimeError(
                        "\nMissing held-out solution:\n"
                        f"  puzzle: {puzzle}\n"
                        f"  expected: {puzzle.stem}_solution.txt\n"
                        f"  searched under: {answer_dir}\n\n"
                        "remainN validation requires the solution file. "
                        "The old full-only testing did not."
                    )

            task_id = (
                f"heldout_{puzzle.stem}:{level}"
            )

            rows.append(
                (
                    task_id,
                    str(puzzle.resolve()),
                    str(solution),
                    prefix,
                )
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w") as f:
        f.write("# task_id\tpuzzle\tsolution\tprefix\n")

        for task_id, puzzle, solution, prefix in rows:
            f.write(
                f"{task_id}\t"
                f"{puzzle}\t"
                f"{solution}\t"
                f"{prefix}\n"
            )

    print(
        f"[heldout] stage={state['stage_index']} "
        f"levels={'+'.join(active_levels)} "
        f"puzzles={len(puzzles)} "
        f"tasks={len(rows)}"
    )
    print(f"[heldout] wrote {out}")


if __name__ == "__main__":
    main()
