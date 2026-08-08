#!/usr/bin/env python3
"""Extract full-board and endgame curriculum metrics into JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def last_float(text: str, label: str, default: float = 0.0) -> float:
    hits = re.findall(rf"\[{re.escape(label)}\]\s+([-+0-9.eE]+)", text)
    return float(hits[-1]) if hits else default


def learner_value(text: str, key: str, default: float = 0.0) -> float:
    hits = re.findall(rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)", text, re.MULTILINE)
    return float(hits[-1]) if hits else default


def piece_count(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    blocks, active = 0, False
    for raw in p.read_text(errors="replace").splitlines() + [""]:
        line = raw.strip()
        if line.startswith("#"):
            continue
        if line:
            active = True
        elif active:
            blocks, active = blocks + 1, False
    return blocks


def sgf_games(sgf: Path) -> list[dict]:
    if not sgf.exists():
        return []
    text = sgf.read_text(errors="replace")
    chunks = re.findall(r"\(;GM\[cornpuzzle\].*?(?=\n\(;GM\[cornpuzzle\]|\Z)", text, re.S)
    games = []
    for chunk in chunks:
        result = re.search(r"RE\[([-+0-9.eE]+)\]", chunk)
        task = re.search(r"CTASK\[([^\]]+)\]", chunk)
        puzzle = re.search(r"PUZZLE\[([^\]]+)\]", chunk)
        if not result or not task:
            continue
        level = task.group(1).rsplit(":", 1)[-1]
        moves = len(re.findall(r";B\[[^\]]*\]", chunk))
        total_pieces = piece_count(puzzle.group(1)) if puzzle else 0
        solved = float(result.group(1)) >= 1.0 - 1e-8
        gap = 0 if solved else max(0, total_pieces - moves) if total_pieces else -1
        games.append({
            "return": float(result.group(1)),
            "solved": solved,
            "level": level,
            "task_id": task.group(1),
            "moves": moves,
            "gap": gap,
        })
    return games


def level_stats(games: list[dict], level: str) -> tuple[float, float, float, int]:
    selected = [g for g in games if g["level"] == level]
    if not selected:
        return 0.0, 0.0, 0.0, 0
    n = len(selected)
    return (
        sum(g["return"] for g in selected) / n,
        sum(g["solved"] for g in selected) / n,
        sum(g["moves"] for g in selected) / n,
        n,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--training-log", required=True)
    p.add_argument("--op-log")
    p.add_argument("--sgf")
    p.add_argument("--iteration-progress", type=float, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    training = Path(args.training_log).read_text(errors="replace")
    op = Path(args.op_log).read_text(errors="replace") if args.op_log and Path(args.op_log).exists() else training
    games = sgf_games(Path(args.sgf)) if args.sgf else []
    full_return, full_solve, full_length, full_n = level_stats(games, "full")
    r2_return, r2_solve, _, r2_n = level_stats(games, "remain2")
    r4_return, r4_solve, _, r4_n = level_stats(games, "remain4")
    r6_return, r6_solve, _, r6_n = level_stats(games, "remain6")
    middle_return, middle_solve, _, middle_n = level_stats(games, "remain10")
    available_endgame = [rate for rate, n in ((r2_solve, r2_n), (r4_solve, r4_n), (r6_solve, r6_n)) if n]
    endgame_solve = sum(available_endgame) / len(available_endgame) if available_endgame else 0.0

    # Completion shaping gives the Teacher a signal before full solve rate moves.
    # A solved game is worth 1; failures one/two/three pieces short receive only
    # 0.25/0.10/0.05 respectively.
    full_games = [g for g in games if g["level"] == "full"]
    def gap_rate(gap: int) -> float:
        return sum((not g["solved"]) and g["gap"] == gap for g in full_games) / len(full_games) if full_games else 0.0
    gap1, gap2, gap3 = gap_rate(1), gap_rate(2), gap_rate(3)
    completion_score = full_solve + 0.25 * gap1 + 0.10 * gap2 + 0.05 * gap3

    task_stats = {}
    for task_id in sorted({g["task_id"] for g in games}):
        selected = [g for g in games if g["task_id"] == task_id]
        task_stats[task_id] = {
            "games": len(selected),
            "solve_rate": sum(g["solved"] for g in selected) / len(selected),
            "mean_return": sum(g["return"] for g in selected) / len(selected),
        }

    result = {
        "iteration_progress": args.iteration_progress,
        "validation_return": full_return if full_n else last_float(training, "SelfPlay Avg. Game Returns"),
        "validation_solve_rate": full_solve,
        "validation_avg_length": full_length if full_n else last_float(training, "SelfPlay Avg. Game Lengths"),
        "endgame_solve_rate": endgame_solve,
        "remain2_solve_rate": r2_solve,
        "remain4_solve_rate": r4_solve,
        "remain6_solve_rate": r6_solve,
        "middle_solve_rate": middle_solve,
        "completion_score": completion_score,
        "gap1_failure_rate": gap1,
        "gap2_failure_rate": gap2,
        "gap3_failure_rate": gap3,
        "near_finish_failure_rate": gap1,
        "validation_games_full": full_n,
        "validation_games_remain2": r2_n,
        "validation_games_remain4": r4_n,
        "validation_games_remain6": r6_n,
        "validation_games_middle": middle_n,
        "task_stats": task_stats,
        "train_return": 0.0,
        "policy_loss": learner_value(op, "loss_policy"),
        "policy_accuracy": learner_value(op, "accuracy_policy"),
        "value_loss": learner_value(op, "loss_value"),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
