#!/usr/bin/env python3
"""Deterministic sliding-window backward curriculum for CornPuzzle."""
from __future__ import annotations
import sys

import argparse
import json
import hashlib
import random
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


TRAIN_VARIANTS = {
    "remain2": 24,
    "remain4": 12,
    "remain6": 8,
    "remain8": 6,
    "remain10": 6,
    "remain12": 4,
    "remain14": 4,
    "remain16": 4,
    "remain18": 4,
    "remain20": 4,
    "remain22": 4,
    "full": 1,
}

def label(levels: tuple[str, ...]) -> str:
    return "+".join(levels)


def load_state(path: Path) -> dict:
    if not path.exists():
        state = {
            "version": 1,
            "stage_index": 0,
            "consecutive_mastery": 0,
            "complete": False,
            "history": [{"stage_index": 0, "label": label(STAGES[0]), "start_iteration": 1}],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2) + "\n")
        return state
    state = json.loads(path.read_text())
    if state.get("version") != 1:
        raise SystemExit("incompatible rule curriculum state; use a new CURRICULUM_DIR")
    return state


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2) + "\n")



def _read_solution_labels(path: Path):
    lines = path.read_text().splitlines()
    labels = set()

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        for token in stripped.split():
            value = int(token)
            if value > 0:
                labels.add(value)

    if not labels:
        raise RuntimeError(f"no labels found in {path}")

    return lines, sorted(labels)


def _write_relabelled_solution(
    source: Path,
    destination: Path,
    omitted_labels,
):
    """
    Existing C++ pre-places solution labels 1..prefix.

    Therefore:
      pre-placed pieces -> low labels
      pieces left for agent -> high labels

    Geometry itself is unchanged.
    """

    lines, labels = _read_solution_labels(source)

    omitted = set(omitted_labels)

    kept = [x for x in labels if x not in omitted]
    left = [x for x in labels if x in omitted]

    remap = {}

    for new_label, old_label in enumerate(kept, 1):
        remap[old_label] = new_label

    for new_label, old_label in enumerate(
        left,
        len(kept) + 1,
    ):
        remap[old_label] = new_label

    out = []

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue

        values = []

        for token in stripped.split():
            value = int(token)

            if value > 0:
                value = remap[value]

            values.append(str(value))

        out.append(" ".join(values))

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        "\n".join(out) + "\n"
    )


def _make_unique_masks(
    labels,
    remain,
    count,
    seed_text,
):
    """
    Deterministic random augmentation:
    same experiment is reproducible, but augXX masks are distinct.
    """

    digest = hashlib.sha256(
        seed_text.encode()
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        "big",
    )

    rng = random.Random(seed)

    masks = []
    seen = set()

    while len(masks) < count:
        mask = tuple(
            sorted(
                rng.sample(labels, remain)
            )
        )

        if mask in seen:
            continue

        seen.add(mask)
        masks.append(mask)

    return masks


def write_manifest(
    task_bank: Path,
    state_path: Path,
    output: Path,
) -> None:

    tasks = json.loads(task_bank.read_text())
    state = load_state(state_path)

    active_levels = STAGES[state["stage_index"]]
    levels = set(active_levels)

    base_tasks = sorted(
        (t for t in tasks if t["level"] in levels),
        key=lambda t: (
            t["level"],
            t["puzzle_id"],
        ),
    )

    expected_base = 80 * len(active_levels)

    if len(base_tasks) != expected_base:
        raise SystemExit(
            f"expected {expected_base} base tasks, "
            f"got {len(base_tasks)}"
        )

    chosen = []

    # Generated solution variants stay inside curriculum dir.
    aug_root = task_bank.parent / "aug_solutions"

    for t in base_tasks:
        level = t["level"]
        variants = TRAIN_VARIANTS[level]

        # Full puzzle does not need augmentation.
        if level == "full":
            chosen.append(dict(t))
            continue

        source_solution = Path(t["solution"])
        _, labels = _read_solution_labels(
            source_solution
        )

        prefix = int(t["prefix"])
        remain = len(labels) - prefix

        expected_remain = int(
            level[len("remain"):]
        )

        if remain != expected_remain:
            raise RuntimeError(
                f"{t['puzzle_id']} {level}: "
                f"labels={len(labels)}, "
                f"prefix={prefix}, "
                f"remain={remain}, "
                f"expected={expected_remain}"
            )

        masks = _make_unique_masks(
            labels,
            remain,
            variants,
            seed_text=(
                f"corn-real-augmentation-v1|"
                f"{t['puzzle_id']}|{level}"
            ),
        )

        root = t["task_id"].rsplit(":", 1)[0]

        for i, omitted in enumerate(masks):
            item = dict(t)

            item["task_id"] = (
                f"{root}:aug{i:02d}:{level}"
            )

            variant_solution = (
                aug_root
                / level
                / (
                    f"{t['puzzle_id']}"
                    f"_aug{i:02d}_solution.txt"
                )
            ).resolve()

            _write_relabelled_solution(
                source_solution,
                variant_solution,
                omitted,
            )

            item["solution"] = str(
                variant_solution
            )

            chosen.append(item)

    expected = 80 * sum(
        TRAIN_VARIANTS[level]
        for level in active_levels
    )

    if len(chosen) != expected:
        raise SystemExit(
            f"expected {expected} augmented tasks, "
            f"got {len(chosen)}"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open("w") as f:
        f.write(
            "# task_id\tpuzzle\tsolution\tprefix\n"
        )

        for t in chosen:
            f.write(
                f"{t['task_id']}\t"
                f"{t['puzzle']}\t"
                f"{t['solution']}\t"
                f"{t['prefix']}\n"
            )

    print(
        f"wrote {len(chosen)} REAL augmented training tasks "
        f"for stage {state['stage_index']} "
        f"({label(active_levels)}): {output}"
    )


def write_level_manifest(task_bank: Path, level: str, output: Path) -> None:
    """Write a fixed 80-task manifest for an evaluation level such as full."""
    tasks = json.loads(task_bank.read_text())
    chosen = sorted(
        (t for t in tasks if t["level"] == level),
        key=lambda t: t["puzzle_id"],
    )
    if len(chosen) != 80:
        raise SystemExit(f"expected 80 tasks for level {level}, got {len(chosen)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("# task_id\tpuzzle\tsolution\tprefix\n")
        for t in chosen:
            f.write(f"{t['task_id']}\t{t['puzzle']}\t{t['solution']}\t{t['prefix']}\n")
    print(f"wrote fixed {level} evaluation manifest with {len(chosen)} tasks: {output}")


def write_folder_manifest(folder: Path, output: Path, expected: int) -> None:
    """Write deterministic full-puzzle tasks for a held-out puzzle folder."""
    puzzles = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == ".txt")
    if len(puzzles) != expected:
        raise SystemExit(f"expected {expected} held-out .txt puzzles in {folder}, got {len(puzzles)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("# task_id\tpuzzle\tsolution\tprefix\n")
        for puzzle in puzzles:
            # solution is unused when prefix=0, but the TSV schema requires it.
            f.write(f"testing_{puzzle.stem}:full\t{puzzle}\t{puzzle}\t0\n")
    print(f"wrote fixed held-out evaluation manifest with {len(puzzles)} tasks: {output}")


def update(args) -> None:
    state_path = Path(args.state)
    state = load_state(state_path)
    metrics = json.loads(Path(args.metrics).read_text())

    levels = STAGES[state["stage_index"]]
    stats = metrics.get("level_stats", {})

    gate_enabled = args.validation_gate.lower() == "true"
    threshold = float(args.validation_solve_rate)

    if not 0.0 <= threshold <= 1.0:
        raise SystemExit(
            "--validation-solve-rate must be between 0 and 1"
        )

    checks = {}
    validation_complete = True
    progression_passed = True

    for level in levels:
        item = stats.get(level, {})

        solve_rate = float(
            item.get("solve_rate", 0.0)
        )

        mean_return = float(
            item.get("mean_return", 0.0)
        )

        covered = int(
            item.get("covered_tasks", 0)
        )

        total = int(
            item.get("total_tasks", 80)
        )

        coverage_complete = (
            covered == total == 80
        )

        threshold_passed = (
            solve_rate >= threshold - 1e-12
        )

        # Gate OFF means validation is measurement-only.
        # Complete coverage is still mandatory so a broken /
        # incomplete validation can never advance curriculum.
        passed = (
            coverage_complete
            and (
                threshold_passed
                if gate_enabled
                else True
            )
        )

        checks[level] = {
            "solve_rate": solve_rate,
            "mean_return": mean_return,
            "covered_tasks": covered,
            "total_tasks": total,
            "coverage_complete": coverage_complete,
            "solve_rate_threshold": threshold,
            "threshold_passed": threshold_passed,
            "passed": passed,
        }

        validation_complete = (
            validation_complete
            and coverage_complete
        )

        progression_passed = (
            progression_passed
            and passed
        )

    advance_now = (
        validation_complete
        and progression_passed
    )

    # Retain this legacy field so existing status/plot code
    # remains compatible. There is no patience requirement now.
    state["consecutive_mastery"] = (
        1 if advance_now else 0
    )

    event = {
        "iteration": args.iteration,
        "stage_index": state["stage_index"],
        "label": label(levels),
        "checks": checks,
        "validation_complete": validation_complete,
        "validation_gate_enabled": gate_enabled,
        "validation_solve_rate_threshold": threshold,
        "mastered_this_round": advance_now,
        "consecutive_mastery": state["consecutive_mastery"],
        "advance_this_round": advance_now,
        "advance_reason": (
            "validation_gate_disabled"
            if advance_now and not gate_enabled
            else (
                "validation_solve_rate"
                if advance_now
                else None
            )
        ),
    }

    state.setdefault(
        "evaluations",
        []
    ).append(event)

    advanced = False

    if advance_now:
        if state["stage_index"] == len(STAGES) - 1:
            state["complete"] = True
            state["history"][-1][
                "end_iteration"
            ] = args.iteration

        else:
            state["history"][-1][
                "end_iteration"
            ] = args.iteration

            state["stage_index"] += 1
            state["consecutive_mastery"] = 0

            state["history"].append({
                "stage_index": state["stage_index"],
                "label": label(
                    STAGES[state["stage_index"]]
                ),
                "start_iteration": args.iteration + 1,
            })

            advanced = True

    save_state(
        state_path,
        state
    )

    print(
        json.dumps(
            event,
            indent=2
        )
    )

    if state["complete"]:
        print(
            "[rule curriculum] "
            "final stage validation complete; "
            "curriculum complete"
        )

    elif advanced:
        msg = (
            "🌈 LEVEL UP! advance to stage "
            f"{state['stage_index']}: "
            f"{label(STAGES[state['stage_index']])} 🌈"
        )

        colors = [
            196, 208, 226, 46,
            51, 21, 93, 201
        ]

        print(
            "".join(
                chr(27)
                + f"[1;38;5;{colors[i % len(colors)]}m"
                + ch
                for i, ch in enumerate(msg)
            )
            + chr(27)
            + "[0m"
            if sys.stdout.isatty()
            else msg
        )

    else:
        if not validation_complete:
            reason = "validation coverage incomplete"

        elif gate_enabled:
            reason = (
                "validation solve-rate gate failed "
                f"(threshold={threshold:.4f})"
            )

        else:
            reason = "validation incomplete"

        msg = (
            "[rule curriculum] remain at stage "
            f"{state['stage_index']}: "
            f"{label(STAGES[state['stage_index']])}; "
            f"{reason}"
        )

        print(
            chr(27)
            + "[1;31m"
            + msg
            + chr(27)
            + "[0m"
            if sys.stdout.isatty()
            else msg
        )


def status(args) -> None:
    state = load_state(Path(args.state))
    result = {
        "stage_index": state["stage_index"],
        "label": label(STAGES[state["stage_index"]]),
        "consecutive_mastery": state["consecutive_mastery"],
        "complete": state["complete"],
    }
    print(json.dumps(result))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write-manifest")
    w.add_argument("--task-bank", required=True)
    w.add_argument("--state", required=True)
    w.add_argument("--output", required=True)
    w.set_defaults(func=lambda a: write_manifest(Path(a.task_bank), Path(a.state), Path(a.output)))
    e = sub.add_parser("write-level-manifest")
    e.add_argument("--task-bank", required=True)
    e.add_argument("--level", required=True)
    e.add_argument("--output", required=True)
    e.set_defaults(func=lambda a: write_level_manifest(Path(a.task_bank), a.level, Path(a.output)))
    f = sub.add_parser("write-folder-manifest")
    f.add_argument("--folder", required=True)
    f.add_argument("--output", required=True)
    f.add_argument("--expected", type=int, default=20)
    f.set_defaults(func=lambda a: write_folder_manifest(Path(a.folder), Path(a.output), a.expected))
    u = sub.add_parser("update")
    u.add_argument("--state", required=True)
    u.add_argument("--metrics", required=True)
    u.add_argument("--iteration", type=int, required=True)
    # Kept for backward CLI compatibility; pure-plateau progression
    # no longer uses repeated mastery/patience.
    u.add_argument("--patience", type=int, default=1)
    u.add_argument(
        "--validation-gate",
        choices=("true", "false"),
        default="true",
    )
    u.add_argument(
        "--validation-solve-rate",
        type=float,
        default=0.95,
    )
    u.set_defaults(func=update)
    s = sub.add_parser("status")
    s.add_argument("--state", required=True)
    s.set_defaults(func=status)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
