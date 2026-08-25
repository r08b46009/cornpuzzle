#!/bin/bash
set -e

RUN_NAME=$1

if [ -z "$RUN_NAME" ]; then
    echo "usage: $0 RUN_NAME"
    exit 1
fi

TESTING3_DIR=/workspace/testing3

LIST=testing3_checkpoint_list.json


if [ ! -f "$LIST" ]; then
    echo "missing $LIST"
    exit 1
fi


python3 - <<PY

import json
import subprocess
import shutil
from pathlib import Path

run=Path("$RUN_NAME")
testing3=Path("$TESTING3_DIR")

items=json.loads(
    Path("$LIST").read_text()
)


for x in items:

    if not x["exists"]:
        continue

    step=x["training_step"]

    out=Path(
        f"Testing3Eval_{run.name}_iter{step}"
    )

    print("="*60)
    print("EVALUATE", step)
    print("="*60)


    if out.exists():
        print("exists:", out)
        continue


    (out/"model").mkdir(
        parents=True
    )

    shutil.copy(
        x["weight"],
        out/"model"/Path(x["weight"]).name
    )


    # copy config
    cfg=list(run.glob("*.cfg"))

    if len(cfg)==0:
        raise RuntimeError("missing cfg")

    shutil.copy(
        cfg[0],
        out/(out.name+".cfg")
    )


    # build testing3 manifest
    subprocess.run(
        [
            "python3",
            "tools/full_eval_manifest.py",
            "--puzzles",
            str(testing3/"puzzles"),
            "--answers",
            str(testing3/"answers"),
            "--output",
            str(out/"testing3_full.tsv")
        ],
        check=True
    )


print("checkpoint preparation finished")

PY
