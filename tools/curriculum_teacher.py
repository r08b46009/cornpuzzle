#!/usr/bin/env python3
"""Endgame-aware contextual-bandit Teacher for CornPuzzle.

The Teacher never solves a puzzle. It chooses answer-derived, guaranteed-solvable
tasks and emphasizes the endgame weakness of a Student that places most pieces
but rarely completes a board.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

def piece_count(path: Path) -> int:
    blocks, active = 0, False
    for raw in path.read_text().splitlines() + [""]:
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line:
            active = True
        elif active:
            blocks, active = blocks + 1, False
    return blocks


def build(args):
    puzzles, answers, tasks = Path(args.puzzles).resolve(), Path(args.answers).resolve(), []
    puzzle_files = sorted(puzzles.glob("*.txt"))
    if len(puzzle_files) != args.expected_puzzles:
        raise SystemExit(f"expected {args.expected_puzzles} puzzles, got {len(puzzle_files)}")
    for puzzle in puzzle_files:
        stem = puzzle.stem
        solution = answers / f"{stem}_solution.txt"
        if not solution.exists():
            raise SystemExit(f"missing answer: {solution}")
        count = piece_count(puzzle)
        # Full puzzle plus a fine curriculum grid every two remaining pieces.
        # We omit remain<count> because it is identical to the full puzzle.
        levels = [("full", count)] + [
            (f"remain{remaining}", remaining)
            for remaining in range(count - (count % 2 or 2), 1, -2)
        ]
        for level, remaining in levels:
            prefix = 0 if level == "full" else count - remaining
            tasks.append({"task_id": f"{stem}:{level}", "puzzle_id": stem,
                          "puzzle": str(puzzle), "solution": str(solution),
                          "level": level, "prefix": prefix, "remaining": remaining})
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tasks, indent=2) + "\n")
    print(f"wrote {len(tasks)} tasks from {len(puzzle_files)} original puzzles: {out}")


def validation(args):
    """Write one fixed diagnostic manifest for full and endgame evaluation."""
    tasks = json.loads(Path(args.task_bank).read_text())
    diagnostic_levels = {"full", "remain2", "remain4", "remain6", "remain10"}
    selected = sorted((t for t in tasks if t["level"] in diagnostic_levels),
                      key=lambda t: (t["level"] != "full", t["remaining"], t["puzzle_id"]))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# task_id\tpuzzle\tsolution\tprefix\n")
        for t in selected:
            f.write(f"{t['task_id']}\t{t['puzzle']}\t{t['solution']}\t{t['prefix']}\n")
    print(f"wrote {len(selected)} fixed diagnostic tasks: {out}")


def context(metrics: dict) -> np.ndarray:
    # Keep every context feature roughly in [0, 1]. The previous raw average
    # length (~20) dominated the LinUCB confidence term.
    bounded_keys = (
        "iteration_progress", "validation_return", "validation_solve_rate",
        "completion_score", "endgame_solve_rate", "remain2_solve_rate",
        "remain4_solve_rate", "remain6_solve_rate", "middle_solve_rate",
        "near_finish_failure_rate", "train_return", "value_loss",
    )
    x = [1.0] + [min(1.0, max(0.0, float(metrics.get(k, 0.0)))) for k in bounded_keys]
    x.extend([
        min(1.0, max(0.0, float(metrics.get("validation_avg_length", 0.0)) / 24.0)),
        min(1.0, max(0.0, float(metrics.get("policy_loss", 0.0)) / 5.0)),
    ])
    return np.asarray(x, dtype=np.float64)


def load_state(path: Path, arms: int, dim: int):
    if path.exists():
        raw = json.loads(path.read_text())
        a, b = np.asarray(raw["a"]), np.asarray(raw["b"])
        if a.shape != (arms, dim, dim) or b.shape != (arms, dim):
            raise SystemExit(
                "Teacher state is from an incompatible curriculum version. "
                "Use a new CURRICULUM_DIR for the new experiment."
            )
        return a, b, raw["counts"], raw.get("history", [])
    return np.tile(np.eye(dim)[None, :, :], (arms, 1, 1)), np.zeros((arms, dim)), [0] * arms, []


def save_state(path, a, b, counts, history, pending=None):
    raw = {"a": a.tolist(), "b": b.tolist(), "counts": counts, "history": history}
    if pending is not None: raw["pending"] = pending
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(raw, indent=2) + "\n")


def take_distinct_puzzles(ranked, tasks, number, used=None):
    """Take task arms greedily while allowing at most one level per puzzle."""
    used = set() if used is None else set(used)
    selected = []
    for arm in ranked:
        puzzle_id = tasks[arm]["puzzle_id"]
        if puzzle_id in used:
            continue
        selected.append(arm); used.add(puzzle_id)
        if len(selected) == number:
            break
    return selected, used


def task_bucket(task: dict) -> str:
    if task["level"] == "full":
        return "full"
    remaining = int(task["remaining"])
    if remaining == 2:
        return "remain2"
    if remaining == 4:
        return "remain4"
    if remaining == 6:
        return "remain6"
    if remaining == 10:
        return "middle"
    return "other"


def curriculum_stage(metrics: dict) -> tuple[str, dict[str, int]]:
    """Increase full-board practice only after prerequisite skills stabilize."""
    endgame = float(metrics.get("endgame_solve_rate", 0.0))
    middle = float(metrics.get("middle_solve_rate", 0.0))
    full = float(metrics.get("validation_solve_rate", 0.0))
    if full >= 0.50:
        return "full80", {"remain2": 1, "remain4": 1, "remain6": 0, "middle": 2, "full": 16}
    if endgame >= 0.85 and middle >= 0.60:
        return "full60", {"remain2": 2, "remain4": 1, "remain6": 1, "middle": 4, "full": 12}
    if endgame >= 0.70:
        return "full40", {"remain2": 3, "remain4": 3, "remain6": 2, "middle": 4, "full": 8}
    return "full20", {"remain2": 4, "remain4": 4, "remain6": 4, "middle": 4, "full": 4}


def select(args):
    rng = random.Random(args.seed)
    tasks = json.loads(Path(args.task_bank).read_text())
    metrics = json.loads(Path(args.metrics).read_text())
    x = context(metrics); state_path = Path(args.state)
    a, b, counts, history = load_state(state_path, len(tasks), len(x))
    if args.strategy == "teacher":
        if args.num_tasks != 20:
            raise SystemExit("endgame-aware Teacher currently requires --num-tasks 20")
        task_stats = metrics.get("task_stats", {})
        scores = []
        for i in range(len(tasks)):
            inv = np.linalg.inv(a[i]); theta = inv @ b[i]
            linucb = float(theta @ x + args.alpha * np.sqrt(x @ inv @ x))
            stat = task_stats.get(tasks[i]["task_id"], {})
            trials = int(stat.get("games", 0))
            success = float(stat.get("solve_rate", 0.5))
            frontier = 1.0 - abs(2.0 * success - 1.0)
            uncertainty = 1.0 / np.sqrt(trials + 1.0)
            scores.append(linucb + 0.30 * frontier + 0.20 * uncertainty + rng.random() * 1e-9)

        stage, quotas = curriculum_stage(metrics)
        # Each band uses LinUCB/frontier choices plus one least-seen exploration
        # arm. Hard quotas prevent the Teacher from avoiding full puzzles.
        chosen, used = [], set()
        for bucket in ("remain2", "remain4", "remain6", "middle", "full"):
            quota = quotas[bucket]
            if quota == 0:
                continue
            pool = [i for i, t in enumerate(tasks) if task_bucket(t) == bucket]
            ranked = sorted(pool, key=lambda i: scores[i], reverse=True)
            exploitation, used = take_distinct_puzzles(ranked, tasks, max(0, quota - 1), used)
            chosen.extend(exploitation)
            exploration_pool = [i for i in pool if tasks[i]["puzzle_id"] not in used]
            exploration_pool.sort(key=lambda i: (counts[i], rng.random()))
            exploratory, used = take_distinct_puzzles(exploration_pool, tasks, 1, used)
            chosen.extend(exploratory)
        learned = list(chosen)
    else:
        puzzle_ids = sorted({t["puzzle_id"] for t in tasks})
        if args.num_tasks > len(puzzle_ids):
            raise SystemExit("num-tasks cannot exceed the number of original puzzles")
        selected_puzzles = rng.sample(puzzle_ids, args.num_tasks)
        chosen = [rng.choice([i for i, t in enumerate(tasks) if t["puzzle_id"] == puzzle_id])
                  for puzzle_id in selected_puzzles]
        learned = []
    chosen = chosen[:args.num_tasks]
    for i in chosen: counts[i] += 1
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# task_id\tpuzzle\tsolution\tprefix\n")
        for i in chosen:
            t = tasks[i]; f.write(f"{t['task_id']}\t{t['puzzle']}\t{t['solution']}\t{t['prefix']}\n")
    pending = {
        "context": x.tolist(), "selected": chosen, "learned": learned,
        "selected_buckets": [task_bucket(tasks[i]) for i in chosen],
        "curriculum_stage": stage if args.strategy == "teacher" else "random",
        "quotas": quotas if args.strategy == "teacher" else {},
        "before_metrics": {
            key: float(metrics.get(key, 0.0))
            for key in ("validation_return", "validation_solve_rate", "completion_score", "endgame_solve_rate")
        },
    }
    save_state(state_path, a, b, counts, history, pending)
    if args.strategy == "teacher":
        print(f"selected {len(chosen)} task types ({args.strategy}, stage={stage}, quotas={quotas}) -> {out}")
    else:
        print(f"selected {len(chosen)} task types ({args.strategy}) -> {out}")


def update(args):
    path = Path(args.state); raw = json.loads(path.read_text()); pending = raw.pop("pending", None)
    if pending is None: raise SystemExit("no pending Teacher selection to update")
    after = json.loads(Path(args.metrics).read_text())
    before = pending.get("before_metrics", {})
    delta_return = float(after.get("validation_return", 0.0)) - float(before.get("validation_return", 0.0))
    delta_full = float(after.get("validation_solve_rate", 0.0)) - float(before.get("validation_solve_rate", 0.0))
    delta_completion = float(after.get("completion_score", 0.0)) - float(before.get("completion_score", 0.0))
    delta_endgame = float(after.get("endgame_solve_rate", 0.0)) - float(before.get("endgame_solve_rate", 0.0))
    reward = 0.50 * delta_full + 0.25 * delta_completion + 0.15 * delta_endgame + 0.10 * delta_return
    a, b = np.asarray(raw["a"]), np.asarray(raw["b"]); x = np.asarray(pending["context"])
    for i in pending["learned"]:
        a[i] += np.outer(x, x); b[i] += reward * x
    history = raw.get("history", [])
    history.append({
        "before_metrics": before,
        "after_metrics": {
            key: float(after.get(key, 0.0))
            for key in ("validation_return", "validation_solve_rate", "completion_score", "endgame_solve_rate")
        },
        "delta_return": delta_return, "delta_full_solve": delta_full, "delta_completion": delta_completion,
        "delta_endgame_solve": delta_endgame, "reward": reward,
        "selected": pending["selected"],
        "selected_buckets": pending.get("selected_buckets", []),
    })
    save_state(path, a, b, raw["counts"], history)
    print(
        "Teacher reward: "
        f"{reward:+.6f} = 0.50*{delta_full:+.6f} (full solve) + "
        f"0.25*{delta_completion:+.6f} (completion) + "
        f"0.15*{delta_endgame:+.6f} (endgame solve) + "
        f"0.10*{delta_return:+.6f} (full return)"
    )


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--puzzles", required=True); b.add_argument("--answers", required=True)
    b.add_argument("--output", required=True); b.add_argument("--expected-puzzles", type=int, default=80); b.set_defaults(func=build)
    v = sub.add_parser("validation"); v.add_argument("--task-bank", required=True)
    v.add_argument("--output", required=True); v.set_defaults(func=validation)
    s = sub.add_parser("select"); s.add_argument("--task-bank", required=True); s.add_argument("--metrics", required=True)
    s.add_argument("--state", required=True); s.add_argument("--output", required=True); s.add_argument("--num-tasks", type=int, default=20)
    s.add_argument("--strategy", choices=("teacher", "random"), default="teacher"); s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0); s.set_defaults(func=select)
    u = sub.add_parser("update"); u.add_argument("--state", required=True); u.add_argument("--metrics", required=True); u.set_defaults(func=update)
    args = p.parse_args(); args.func(args)


if __name__ == "__main__": main()
