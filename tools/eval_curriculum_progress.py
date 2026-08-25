#!/usr/bin/env python3

import json
from pathlib import Path
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--run", required=True)
parser.add_argument("--state", required=True)

args = parser.parse_args()


run_dir = Path(args.run)

state = json.loads(
    Path(args.state).read_text()
)


print("Found curriculum checkpoints:")
print("--------------------------------")


for h in state["history"]:

    iteration = h["end_iteration"]

    # same as step_for()
    step = iteration * 500

    weight = (
        run_dir
        / "model"
        / f"weight_iter_{step}.pt"
    )

    print(
f"""
LEVEL:
 {h['label']}

curriculum iteration:
 {iteration}

training step:
 {step}

checkpoint:
 {weight}

exists:
 {weight.exists()}
"""
    )
