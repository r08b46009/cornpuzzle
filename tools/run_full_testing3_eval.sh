#!/bin/bash
set -e

RUN_NAME=$1
WEIGHT_ITER=$2

TESTING3_DIR=/workspace/testing3
OUT_NAME="Testing3Full_${RUN_NAME}_iter${WEIGHT_ITER}"

echo "===== Full puzzle evaluation ====="
echo "RUN=${RUN_NAME}"
echo "ITER=${WEIGHT_ITER}"

mkdir -p ${OUT_NAME}/model ${OUT_NAME}/sgf

# config
cp ${RUN_NAME}/${RUN_NAME}.cfg \
   ${OUT_NAME}/${OUT_NAME}.cfg

# weights
cp ${RUN_NAME}/model/weight_iter_${WEIGHT_ITER}.pt \
   ${OUT_NAME}/model/

cp ${RUN_NAME}/model/weight_iter_${WEIGHT_ITER}.pkl \
   ${OUT_NAME}/model/


# build manifest
python3 tools/full_eval_manifest.py \
    --puzzles ${TESTING3_DIR}/puzzles \
    --answers ${TESTING3_DIR}/answers \
    --output ${OUT_NAME}/testing3_full.tsv


TASKS=$(python3 - <<PY
from pathlib import Path
n=0
for x in Path("${OUT_NAME}/testing3_full.tsv").read_text().splitlines():
    if x and not x.startswith("#"):
        n+=1
print(n)
PY
)

echo "tasks=${TASKS}"


COUNTER=${OUT_NAME}/counter
echo 0 > ${COUNTER}


tools/quick-run.sh train cornpuzzle \
    ${OUT_NAME}/${OUT_NAME}.cfg \
    1 \
    -n ${OUT_NAME} \
    --sp_gpu 0 \
    --op_gpu -1 \
    -b 32 \
    -p 22803 \
    --sp_progress \
    -conf_str "actor_num_simulation=50:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_use_dirichlet_noise=false:actor_use_gumbel_noise=false:zero_num_threads=1:zero_num_parallel_games=1:env_cornpuzzle_curriculum_enable=true:env_cornpuzzle_curriculum_tasks_file=${OUT_NAME}/testing3_full.tsv:env_cornpuzzle_curriculum_sequential=true:env_cornpuzzle_curriculum_counter_file=${COUNTER}:env_compound_puzzles_dir=${TESTING3_DIR}/puzzles:zero_num_games_per_iteration=${TASKS}:learner_training_step=0"


python3 tools/rule_curriculum_metrics.py \
    --sgf ${OUT_NAME}/sgf/1.sgf \
    --manifest ${OUT_NAME}/testing3_full.tsv \
    --training-log ${OUT_NAME}/Training.log \
    --op-log ${OUT_NAME}/op.log \
    --iteration ${WEIGHT_ITER} \
    --output ${OUT_NAME}/metrics.json


echo "===== DONE ====="
cat ${OUT_NAME}/metrics.json

