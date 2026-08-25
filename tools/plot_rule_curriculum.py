#!/usr/bin/env python3
"""Continuously update training curves with curriculum-stage boundaries."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rule-curriculum")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def timestamp(text: str):
    match = re.search(r"\[(\d{4}/\d{2}/\d{2}[_ ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]", text)
    if not match:
        return None
    raw = match.group(1).replace(" ", "_")
    return datetime.strptime(raw, "%Y/%m/%d_%H:%M:%S.%f")


def stage_boundaries(state: dict, learner: bool) -> list[tuple[float, str]]:
    result = []
    for item in state.get("history", []):
        iteration = int(item["start_iteration"])
        x = (iteration - 1) * 500 if learner else iteration
        result.append((x, item["label"]))
    return result


def decorate(ax, boundaries, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    colors = ("#edf6ff", "#fff4e6")
    x_right = ax.get_xlim()[1]
    for i, (x, stage) in enumerate(boundaries):
        next_x = boundaries[i + 1][0] if i + 1 < len(boundaries) else x_right
        ax.axvspan(x, next_x, color=colors[i % 2], alpha=0.32, zorder=0)
        if i > 0:
            ax.axvline(x, color="black", linestyle="--", linewidth=1.2, alpha=0.65)
        ax.text(x, 1.01, stage, rotation=35, ha="left", va="bottom",
                transform=ax.get_xaxis_transform(), fontsize=8)


def save_curve(output: Path, name: str, xs, series, boundaries, xlabel, ylabel):
    if not xs or not any(values for _, values in series):
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, values in series:
        n = min(len(xs), len(values))
        if n:
            ax.plot(xs[:n], values[:n], linewidth=2, label=label)
    decorate(ax, boundaries, f"{ylabel} with rule-based curriculum stages", xlabel, ylabel)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    plt.close(fig)


def learner_curves(op: str):
    rows = []
    pattern = re.compile(
        r"nn step\s+(\d+),.*?loss_policy:\s*([-+0-9.eE]+).*?"
        r"accuracy_policy:\s*([-+0-9.eE]+).*?loss_value:\s*([-+0-9.eE]+)",
        re.S,
    )
    for step, policy, accuracy, value in pattern.findall(op):
        rows.append((int(step), float(policy), float(accuracy), float(value)))
    # A retry can log the same step twice; retain the latest value.
    return sorted({row[0]: row for row in rows}.values())


def selfplay_curves(training: str):
    chunks = re.findall(
        r"(\[[^\n]+\]\s+\[Iteration\]\s*=+(\d+)=+.*?)(?=\[[^\n]+\]\s+\[Iteration\]\s*=+\d+=+|\Z)",
        training,
        re.S,
    )
    result = {}
    for chunk, iteration_raw in chunks:
        if "[SelfPlay] Finished." not in chunk:
            continue
        iteration = int(iteration_raw)
        def metric(label):
            hits = re.findall(rf"\[{re.escape(label)}\]\s+([-+0-9.eE]+)", chunk)
            return float(hits[-1]) if hits else None
        start_line = re.search(r"\[[^\n]+\]\s+\[Iteration\]", chunk)
        op_start_line = re.search(r"\[[^\n]+\]\s+\[Optimization\]\s+Start", chunk)
        op_finish_line = re.search(r"\[[^\n]+\]\s+\[Optimization\]\s+Finished", chunk)
        start = timestamp(start_line.group(0)) if start_line else None
        op_start = timestamp(op_start_line.group(0)) if op_start_line else None
        finish = timestamp(op_finish_line.group(0)) if op_finish_line else None
        result[iteration] = {
            "return_min": metric("SelfPlay Min. Game Returns"),
            "return_avg": metric("SelfPlay Avg. Game Returns"),
            "return_max": metric("SelfPlay Max. Game Returns"),
            "length_min": metric("SelfPlay Min. Game Lengths"),
            "length_avg": metric("SelfPlay Avg. Game Lengths"),
            "length_max": metric("SelfPlay Max. Game Lengths"),
            "sp_time": (op_start - start).total_seconds() if start and op_start else None,
            "op_time": (finish - op_start).total_seconds() if finish and op_start else None,
            "total_time": (finish - start).total_seconds() if finish and start else None,
        }
    return result


def testing_metrics_from_folders(root: Path, run_name: str):
    """Read held-out complete-solve accuracy and reward from TestingRule folders."""
    result = {}
    prefix = f"TestingRule_{run_name}_iter"
    for folder in root.glob(f"{prefix}*"):
        if not folder.is_dir():
            continue
        suffix = folder.name[len(prefix):]
        if not suffix.isdigit():
            continue
        sgf = folder / "sgf" / "1.sgf"
        if not sgf.exists():
            continue
        text = sgf.read_text(errors="replace")
        games = re.findall(r"\(;GM\[cornpuzzle\].*?(?=\n\(;GM\[cornpuzzle\]|\Z)", text, re.S)
        returns = []
        for game in games:
            match = re.search(r"RE\[([-+0-9.eE]+)\]", game)
            if match:
                returns.append(float(match.group(1)))
        if returns:
            result[int(suffix)] = {
                "solve_rate": sum(value >= 1.0 - 1e-8 for value in returns) / len(returns),
                "reward": sum(returns) / len(returns),
                "games": len(returns),
            }
    return result


def evaluation_metrics_from_json_dir(root: Path):
    """Aggregate per-level evaluation JSON into one result per iteration."""
    result = {}
    if not root.exists():
        return result
    for path in root.glob("after_iter*.json"):
        match = re.fullmatch(r"after_iter(\d+)\.json", path.name)
        if not match:
            continue
        payload = json.loads(path.read_text())
        levels = payload.get("level_stats", {})
        rows = [row for row in levels.values() if int(row.get("games", 0)) > 0]
        games = sum(int(row["games"]) for row in rows)
        if not games:
            continue
        result[int(match.group(1))] = {
            "solve_rate": sum(float(row["solve_rate"]) * int(row["games"])
                              for row in rows) / games,
            "reward": sum(float(row["mean_return"]) * int(row["games"])
                          for row in rows) / games,
            "games": games,
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--curriculum-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--testing-root", help="directory containing TestingRule_<run>_iter<N> folders")
    p.add_argument("--full80-metrics-dir",
                   help="directory containing deterministic Full-80 metric JSON files")
    args = p.parse_args()
    run = Path(args.run_dir)
    curriculum = Path(args.curriculum_dir)
    output = Path(args.output_dir)
    testing_root = Path(args.testing_root) if args.testing_root else run.resolve().parent
    full80_metrics_dir = (Path(args.full80_metrics_dir) if args.full80_metrics_dir
                          else curriculum / "full80_metrics")
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((curriculum / "rule_state.json").read_text())
    op = (run / "op.log").read_text(errors="replace")
    training = (run / "Training.log").read_text(errors="replace")

    learner = learner_curves(op)
    learner_x = [row[0] for row in learner]
    learner_bounds = stage_boundaries(state, True)
    save_curve(output, "accuracy_policy", learner_x,
               [("accuracy_policy", [r[2] for r in learner])], learner_bounds,
               "NN steps", "Policy accuracy")
    save_curve(output, "loss_policy", learner_x,
               [("loss_policy", [r[1] for r in learner])], learner_bounds,
               "NN steps", "Policy loss")
    save_curve(output, "loss_value", learner_x,
               [("loss_value", [r[3] for r in learner])], learner_bounds,
               "NN steps", "Value loss")

    sp = selfplay_curves(training)
    iterations = sorted(sp)
    iter_bounds = stage_boundaries(state, False)
    def values(key):
        return [sp[i][key] for i in iterations]
    save_curve(output, "returns", iterations,
               [("Min", values("return_min")), ("Avg", values("return_avg")), ("Max", values("return_max"))],
               iter_bounds, "Iteration", "Game return")
    save_curve(output, "lengths", iterations,
               [("Min", values("length_min")), ("Avg", values("length_avg")), ("Max", values("length_max"))],
               iter_bounds, "Iteration", "Game length")
    save_curve(output, "time", iterations,
               [("Self-play", values("sp_time")), ("Optimization", values("op_time")), ("Total", values("total_time"))],
               iter_bounds, "Iteration", "Seconds")

    # Only the deterministic current-stage mastery result controls promotion.
    # Full-80 and held-out results are monitoring signals and never update state.
    mastery = evaluation_metrics_from_json_dir(curriculum / "metrics")
    mastery_x = sorted(mastery)
    save_curve(output, "curriculum_solve_rate", mastery_x,
               [("Current-stage deterministic solve rate",
                 [mastery[i]["solve_rate"] for i in mastery_x])],
               iter_bounds, "Iteration", "Curriculum solve rate")
    save_curve(output, "curriculum_reward", mastery_x,
               [("Current-stage average return",
                 [mastery[i]["reward"] for i in mastery_x])],
               iter_bounds, "Iteration", "Curriculum reward")

    full80 = evaluation_metrics_from_json_dir(full80_metrics_dir)
    full80_x = sorted(full80)
    save_curve(output, "full80_solve_rate", full80_x,
               [("In-domain Full-80 solve rate",
                 [full80[i]["solve_rate"] for i in full80_x])],
               iter_bounds, "Iteration", "Full-80 solve rate")
    save_curve(output, "full80_reward", full80_x,
               [("In-domain Full-80 average return",
                 [full80[i]["reward"] for i in full80_x])],
               iter_bounds, "Iteration", "Full-80 reward")

    testing = testing_metrics_from_folders(testing_root, run.name)
    testing_x = sorted(testing)
    save_curve(output, "testing_solve_rate", testing_x,
               [("Held-out complete-solve rate",
                 [testing[i]["solve_rate"] for i in testing_x])],
               iter_bounds, "Iteration", "Testing solve rate")
    save_curve(output, "testing_reward", testing_x,
               [("Held-out average game return",
                 [testing[i]["reward"] for i in testing_x])],
               iter_bounds, "Iteration", "Testing reward")

    (output / "latest.json").write_text(json.dumps({
        "latest_iteration": max(iterations) if iterations else 0,
        "stage_index": state["stage_index"],
        "stage": state["history"][-1]["label"],
        "files": ["accuracy_policy.png", "loss_policy.png", "loss_value.png",
                  "returns.png", "lengths.png", "time.png",
                  "curriculum_solve_rate.png", "curriculum_reward.png",
                  "full80_solve_rate.png", "full80_reward.png",
                  "testing_solve_rate.png", "testing_reward.png"],
    }, indent=2) + "\n")
    print(f"updated curriculum plots: {output}")


if __name__ == "__main__":
    main()
