#!/usr/bin/env python3
"""Deterministic sliding-window backward curriculum for CornPuzzle."""
from __future__ import annotations

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


def write_manifest(task_bank: Path, state_path: Path, output: Path) -> None:
    tasks = json.loads(task_bank.read_text())
    state = load_state(state_path)
    levels = set(STAGES[state["stage_index"]])
    chosen = sorted(
        (t for t in tasks if t["level"] in levels),
        key=lambda t: (t["level"], t["puzzle_id"]),
    )
    expected = 80 * len(levels)
    if len(chosen) != expected:
        raise SystemExit(f"expected {expected} tasks for {sorted(levels)}, got {len(chosen)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("# task_id\tpuzzle\tsolution\tprefix\n")
        for t in chosen:
            f.write(f"{t['task_id']}\t{t['puzzle']}\t{t['solution']}\t{t['prefix']}\n")
    print(f"wrote {len(chosen)} tasks for stage {state['stage_index']} ({label(STAGES[state['stage_index']])}): {output}")


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

    checks = {}
    mastered = True
    for level in levels:
        item = stats.get(level, {})
        solve_rate = float(item.get("solve_rate", 0.0))
        mean_return = float(item.get("mean_return", 0.0))
        covered = int(item.get("covered_tasks", 0))
        total = int(item.get("total_tasks", 80))
        passed = (solve_rate >= 1.0 - 1e-12 and
                  mean_return >= 1.0 - 1e-12 and
                  covered == total == 80)
        checks[level] = {
            "solve_rate": solve_rate,
            "mean_return": mean_return,
            "covered_tasks": covered,
            "total_tasks": total,
            "passed": passed,
        }
        mastered = mastered and passed

    state["consecutive_mastery"] = state["consecutive_mastery"] + 1 if mastered else 0
    event = {
        "iteration": args.iteration,
        "stage_index": state["stage_index"],
        "label": label(levels),
        "checks": checks,
        "mastered_this_round": mastered,
        "consecutive_mastery": state["consecutive_mastery"],
    }
    state.setdefault("evaluations", []).append(event)

    advanced = False
    if state["consecutive_mastery"] >= args.patience:
        if state["stage_index"] == len(STAGES) - 1:
            state["complete"] = True
            state["history"][-1]["end_iteration"] = args.iteration
        else:
            state["history"][-1]["end_iteration"] = args.iteration
            state["stage_index"] += 1
            state["consecutive_mastery"] = 0
            state["history"].append({
                "stage_index": state["stage_index"],
                "label": label(STAGES[state["stage_index"]]),
                "start_iteration": args.iteration + 1,
            })
            advanced = True

    save_state(state_path, state)
    print(json.dumps(event, indent=2))
    if state["complete"]:
        print("[rule curriculum] full-only stage mastered at 100%; curriculum complete")
    elif advanced:
        print(f"[rule curriculum] advance to stage {state['stage_index']}: {label(STAGES[state['stage_index']])}")
    else:
        print(f"[rule curriculum] remain at stage {state['stage_index']}: {label(STAGES[state['stage_index']])}")


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
    u.add_argument("--patience", type=int, default=2)
    u.set_defaults(func=update)
    s = sub.add_parser("status")
    s.add_argument("--state", required=True)
    s.set_defaults(func=status)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
