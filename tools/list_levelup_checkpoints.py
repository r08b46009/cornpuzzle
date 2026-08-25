#!/usr/bin/env python3

import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--state", required=True)

args = parser.parse_args()

run = Path(args.run)

state=json.loads(Path(args.state).read_text())


# 依照你的 step_for
# 先不要猜，之後改成實際函數
for h in state["history"]:
    iteration=h["end_iteration"]

    print(
        f"{h['label']:20s} curriculum_iter={iteration}"
    )
