#!/usr/bin/env python3

import json
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run",
        required=True
    )

    parser.add_argument(
        "--step-factor",
        type=int,
        default=500
    )

    args = parser.parse_args()

    run = Path(args.run)

    state_file = run / "curriculum" / "rule_state.json"

    if not state_file.exists():
        raise RuntimeError(
            f"Missing {state_file}"
        )

    state = json.loads(
        state_file.read_text()
    )

    output=[]

    for h in state["history"]:

        curriculum_iter = h["end_iteration"]

        step = curriculum_iter * args.step_factor

        weight = (
            run /
            "model" /
            f"weight_iter_{step}.pt"
        )

        item={
            "level": h["label"],
            "curriculum_iteration": curriculum_iter,
            "training_step": step,
            "weight": str(weight),
            "exists": weight.exists()
        }

        output.append(item)

        print(
            f"{item['level']:20s} "
            f"iter={curriculum_iter:4d} "
            f"step={step:6d} "
            f"exists={weight.exists()}"
        )


    out=Path(
        "testing3_checkpoint_list.json"
    )

    out.write_text(
        json.dumps(
            output,
            indent=2
        )
    )

    print()
    print(
        "written:",
        out
    )


if __name__=="__main__":
    main()
