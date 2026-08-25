#!/bin/bash
set -e

RUN_NAME=$1

if [ -z "$RUN_NAME" ]; then
    echo "usage: bash tools/run_testing3_curriculum_all.sh RUN_NAME"
    exit 1
fi

BASE=${RUN_NAME}
PUZZLES=/workspace/testing3/puzzles
ANSWERS=/workspace/testing3/answers

declare -a LEVELS=(
"remain2 7 3500"
"remain2+remain4 16 8000"
"remain4+remain6 26 13000"
"remain6+remain8 45 22500"
"remain8+remain10 67 33500"
"remain10+remain12 85 42500"
"remain12+remain14 97 48500"
"remain14+remain16 145 72500"
"remain16+remain18 164 82000"
"remain18+remain20 179 89500"
"remain20+remain22 194 97000"
"remain22+full 209 104500"
"full 224 112000"
)

for item in "${LEVELS[@]}"; do
    read LEVEL CURRIC_ITER WEIGHT <<< "$item"

    OUT="Testing3Full_${BASE}_iter${WEIGHT}"

    echo "===================================="
    echo "LEVEL=${LEVEL}"
    echo "WEIGHT=${WEIGHT}"
    echo "===================================="

    if [ -f "${OUT}/metrics.json" ]; then
        echo "already finished ${OUT}"
        continue
    fi

    rm -rf "${OUT}"
    mkdir -p "${OUT}/model" "${OUT}/sgf"

    cp "${BASE}/${BASE}.cfg" "${OUT}/${OUT}.cfg"
    cp "${BASE}/model/weight_iter_${WEIGHT}.pt" "${OUT}/model/"
    cp "${BASE}/model/weight_iter_${WEIGHT}.pkl" "${OUT}/model/"

    python3 tools/full_eval_manifest.py \
        --puzzles ${PUZZLES} \
        --answers ${ANSWERS} \
        --output ${OUT}/testing3_full.tsv

    TASKS=$(grep -vc '^#\|^$' ${OUT}/testing3_full.tsv)

    echo "tasks=${TASKS}"

    echo 0 > ${OUT}/counter

    tools/quick-run.sh train cornpuzzle \
        ${OUT}/${OUT}.cfg \
        1 \
        -n ${OUT} \
        --sp_gpu 0 \
        --op_gpu 2 \
        -b 32 \
        -p 22803 \
        --sp_progress \
        -conf_str "nn_file_name=${OUT}/model/weight_iter_${WEIGHT}.pt:actor_num_simulation=50:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_use_dirichlet_noise=false:actor_use_gumbel_noise=false:zero_num_threads=1:zero_num_parallel_games=1:zero_num_games_per_iteration=${TASKS}:env_cornpuzzle_curriculum_enable=true:env_cornpuzzle_curriculum_tasks_file=${OUT}/testing3_full.tsv:env_cornpuzzle_curriculum_sequential=true:env_cornpuzzle_curriculum_counter_file=${OUT}/counter:env_compound_puzzles_dir=${PUZZLES}:learner_training_step=0"

    python3 tools/rule_curriculum_metrics.py \
        --sgf ${OUT}/sgf/1.sgf \
        --manifest ${OUT}/testing3_full.tsv \
        --training-log ${OUT}/Training.log \
        --op-log ${OUT}/op.log \
        --iteration ${WEIGHT} \
        --output ${OUT}/metrics.json

done
