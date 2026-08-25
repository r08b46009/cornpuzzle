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


def label(levels):
    return "+".join(levels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--metrics", required=True)
    args = p.parse_args()

    state_path = Path(args.state)
    state = json.loads(state_path.read_text())

    metrics_path = Path(args.metrics)
    metrics = json.loads(metrics_path.read_text())

    old_stage = int(state["stage_index"])
    old_levels = STAGES[old_stage]

    measurement = {
        "iteration": args.iteration,
        "stage_index": old_stage,
        "label": label(old_levels),
        "reason": "training_plateau",
        "heldout_measurement_only": True,
        "level_stats": metrics.get("level_stats", {}),
    }

    state.setdefault(
        "heldout_measurements",
        []
    ).append(measurement)

    # Close current stage.
    if state.get("history"):
        state["history"][-1]["end_iteration"] = args.iteration

    state["consecutive_mastery"] = 0

    if old_stage == len(STAGES) - 1:
        state["complete"] = True

        print(
            "[curriculum] final full stage plateau reached; "
            "held-out measurement recorded; curriculum complete"
        )

    else:
        new_stage = old_stage + 1
        state["stage_index"] = new_stage

        state.setdefault("history", []).append({
            "stage_index": new_stage,
            "label": label(STAGES[new_stage]),
            "start_iteration": args.iteration + 1,
        })

        print(
            f"[curriculum] ADVANCE "
            f"{old_stage}:{label(old_levels)} "
            f"-> "
            f"{new_stage}:{label(STAGES[new_stage])}"
        )

    state_path.write_text(
        json.dumps(state, indent=2) + "\n"
    )

    print("[curriculum] held-out measurement:")
    print(
        json.dumps(
            metrics.get("level_stats", {}),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
