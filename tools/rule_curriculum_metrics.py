#!/usr/bin/env python3
"""Measure strict per-task mastery for the rule-based curriculum."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_games(path: Path) -> list[dict]:
    text = path.read_text(errors="replace")
    chunks = re.findall(r"\(;GM\[cornpuzzle\].*?(?=\n\(;GM\[cornpuzzle\]|\Z)", text, re.S)
    games = []
    for chunk in chunks:
        result = re.search(r"RE\[([-+0-9.eE]+)\]", chunk)
        task = re.search(r"CTASK\[([^\]]+)\]", chunk)
        if not result or not task:
            continue
        task_id = task.group(1)
        games.append({
            "task_id": task_id,
            "level": task_id.rsplit(":", 1)[-1],
            "return": float(result.group(1)),
            "solved": float(result.group(1)) >= 1.0 - 1e-8,
            "moves": len(re.findall(r";B\[[^\]]*\]", chunk)),
        })
    return games


def expected_tasks(manifest: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for raw in manifest.read_text().splitlines():
        if not raw or raw.startswith("#"):
            continue
        task_id = raw.split("\t", 1)[0]
        result[task_id.rsplit(":", 1)[-1]].add(task_id)
    return result


def last(pattern: str, text: str, default: float = 0.0) -> float:
    hits = re.findall(pattern, text, re.M | re.S)
    return float(hits[-1]) if hits else default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sgf", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--training-log", required=True)
    p.add_argument("--op-log", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    games = parse_games(Path(args.sgf))
    expected = expected_tasks(Path(args.manifest))
    by_level: dict[str, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for game in games:
        by_level[game["level"]].append(game)
        by_task[game["task_id"]].append(game)

    level_stats = {}
    for level, task_ids in expected.items():
        selected = by_level.get(level, [])
        covered = {g["task_id"] for g in selected}
        level_stats[level] = {
            "games": len(selected),
            "solve_rate": sum(g["solved"] for g in selected) / len(selected) if selected else 0.0,
            "mean_return": sum(g["return"] for g in selected) / len(selected) if selected else 0.0,
            "avg_length": sum(g["moves"] for g in selected) / len(selected) if selected else 0.0,
            "covered_tasks": len(covered & task_ids),
            "total_tasks": len(task_ids),
            "all_tasks_solved": all(
                task_id in by_task and all(g["solved"] for g in by_task[task_id])
                for task_id in task_ids
            ),
        }

    training = Path(args.training_log).read_text(errors="replace")
    op = Path(args.op_log).read_text(errors="replace")
    blocks = re.findall(
        rf"\[Iteration\]\s*=+{args.iteration}=+(.*?)(?=\[Iteration\]\s*=+\d+=+|\Z)",
        training,
        re.S,
    )
    block = blocks[-1] if blocks else ""
    target_step = args.iteration * 500
    step_match = re.search(rf"nn step {target_step},.*?(?=nn step \d+,|\[command\]|\Z)", op, re.S)
    step_block = step_match.group(0) if step_match else ""

    result = {
        "iteration": args.iteration,
        "nn_step": target_step,
        "level_stats": level_stats,
        "train_return": last(r"\[SelfPlay Avg\. Game Returns\]\s+([-+0-9.eE]+)", block),
        "train_avg_length": last(r"\[SelfPlay Avg\. Game Lengths\]\s+([-+0-9.eE]+)", block),
        "policy_loss": last(r"loss_policy:\s*([-+0-9.eE]+)", step_block),
        "policy_accuracy": last(r"accuracy_policy:\s*([-+0-9.eE]+)", step_block),
        "value_loss": last(r"loss_value:\s*([-+0-9.eE]+)", step_block),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
