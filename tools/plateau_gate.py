#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def slope(values):
    n = len(values)
    xs = list(range(n))
    xm = sum(xs) / n
    ym = sum(values) / n

    den = sum((x - xm) ** 2 for x in xs)
    if den == 0:
        return 0.0

    return sum(
        (x - xm) * (y - ym)
        for x, y in zip(xs, values)
    ) / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-log", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--slope-threshold", type=float, default=0.001)
    p.add_argument("--min-iterations", type=int, default=8)
    args = p.parse_args()

    state = json.loads(Path(args.state).read_text())

    history = state.get("history", [])
    if not history:
        print("[plateau] no stage history")
        sys.exit(1)

    stage_start = int(history[-1]["start_iteration"])

    text = Path(args.training_log).read_text(errors="replace")

    # Extract each MiniZero iteration block and its final
    # SelfPlay Avg. Game Returns.
    blocks = re.findall(
        r"\[Iteration\]\s*=+(\d+)=+(.*?)(?=\[Iteration\]\s*=+\d+=+|\Z)",
        text,
        re.S,
    )

    rows = []

    for iteration_text, block in blocks:
        iteration = int(iteration_text)

        if iteration < stage_start:
            continue

        hits = re.findall(
            r"\[SelfPlay Avg\. Game Returns\]\s+([-+0-9.eE]+)",
            block,
        )

        if hits:
            rows.append((iteration, float(hits[-1])))

    if len(rows) < args.min_iterations:
        print(
            f"[plateau] stage samples={len(rows)} "
            f"< min_iterations={args.min_iterations}; continue training"
        )
        sys.exit(1)

    if len(rows) < args.window:
        print(
            f"[plateau] stage samples={len(rows)} "
            f"< window={args.window}; continue training"
        )
        sys.exit(1)

    recent = rows[-args.window:]
    values = [v for _, v in recent]
    m = slope(values)

    print(
        "[plateau] iterations="
        + ",".join(str(i) for i, _ in recent)
    )
    print(
        "[plateau] returns="
        + ",".join(f"{v:.6f}" for v in values)
    )
    print(
        f"[plateau] slope={m:.8f} "
        f"abs={abs(m):.8f} "
        f"threshold={args.slope_threshold:.8f}"
    )

    if abs(m) <= args.slope_threshold:
        print("[plateau] REACHED -> held-out measurement + advance")
        sys.exit(0)

    print("[plateau] NOT reached -> continue training")
    sys.exit(1)


if __name__ == "__main__":
    main()
