#!/bin/bash
set -e

ROOT="Testing3_Evaluation_0816_plateau_aug"
LIST="testing3_checkpoint_list.json"

mkdir -p "$ROOT"

python3 - <<'PY'
import json
import subprocess
import shutil
from pathlib import Path

root = Path("Testing3_Evaluation_0816_plateau_aug")
items = json.loads(Path("testing3_checkpoint_list.json").read_text())

for x in items:

    if not x["exists"]:
        continue

    step = x["training_step"]
    weight = Path(x["weight"])

    out = root / f"iter{step}"
    sgf = out / "sgf"

    print("="*60)
    print("EVALUATE", step)
    print("="*60)

    out.mkdir(parents=True, exist_ok=True)
    sgf.mkdir(exist_ok=True)

    # copy weight
    shutil.copy(
        weight,
        out / weight.name
    )

    # counter
    counter = out / "counter"
    counter.write_text("0")

    cfg = list(Path(".").glob(
        "Testing3Eval_*/*.cfg"
    ))

    if len(cfg) == 0:
        raise RuntimeError("cannot find cfg")

    shutil.copy(
        cfg[0],
        out / f"iter{step}.cfg"
    )

    conf = [
        "nn_file_name=" + str(out / weight.name),
        "zero_training_directory=" + str(out),
        "zero_num_threads=1",
        "zero_num_parallel_games=32",
        "actor_num_simulation=50",
        "actor_select_action_by_count=true",
        "actor_use_dirichlet_noise=false",
        "actor_use_gumbel_noise=false",

        "env_compound_puzzles_dir=/workspace/testing3/puzzles",

        "env_cornpuzzle_curriculum_enable=true",
        f"env_cornpuzzle_curriculum_tasks_file={out}/testing3_full.tsv",
        "env_cornpuzzle_curriculum_sequential=true",
        f"env_cornpuzzle_curriculum_counter_file={counter}"
    ]

    # copy manifest
    shutil.copy(
        "testing3_full.tsv",
        out / "testing3_full.tsv"
    )

    cmd=[
        "build/cornpuzzle/minizero_cornpuzzle",
        "-mode",
        "sp",
        "-conf_file",
        str(out / f"iter{step}.cfg"),
        "-conf_str",
        ":".join(conf)
    ]

    subprocess.run(cmd, check=True)

print("SELF PLAY FINISHED")
PY


python3 tools/testing3_collect_metrics.py \
    --root "$ROOT"

