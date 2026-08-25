#!/bin/bash
set -e

RUN=$1
STATE=${RUN}/curriculum/rule_state.json

if [ -z "$RUN" ]; then
    echo "Usage: bash tools/run_testing3_curriculum_all.sh RUN_NAME"
    exit 1
fi

python3 - <<PY > /tmp/${RUN}_curriculum_steps.txt
import json

state=json.load(open("${STATE}"))

for h in state["history"]:
    print(h["label"], h["end_iteration"]*500)

PY

while read LABEL STEP
do
    echo "===================================="
    echo "LEVEL=${LABEL}"
    echo "WEIGHT=${STEP}"
    echo "===================================="

    bash tools/run_full_testing3_eval.sh \
        ${RUN} \
        ${STEP}

done < /tmp/${RUN}_curriculum_steps.txt

echo "ALL TESTING3 EVALUATIONS DONE"
