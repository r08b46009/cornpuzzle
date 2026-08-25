#!/bin/bash
set -e

for d in Testing3Eval_0816_plateau_aug_iter*
do
    echo "===================================="
    echo "RUNNING $d"
    echo "===================================="

    COUNTER=$d/counter
    echo 0 > $COUNTER
    cfg_d=TestingRule_RuleCurriculum_seed0_iter2/TestingRule_RuleCurriculum_seed0_iter2
    if ! tools/quick-run.sh train cornpuzzle \
        $cfg_d.cfg \
        1 \
        -n $d \
        --sp_gpu 2 \
        --op_gpu 2 \
        -b 32 \
        -p 22803 \
        --sp_progress \
        -conf_str "actor_num_simulation=50:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_use_dirichlet_noise=false:actor_use_gumbel_noise=false:zero_num_threads=1:zero_num_parallel_games=32:env_cornpuzzle_curriculum_enable=true:env_cornpuzzle_curriculum_tasks_file=$d/testing3_full.tsv:env_cornpuzzle_curriculum_sequential=true:env_cornpuzzle_curriculum_counter_file=$COUNTER:env_compound_puzzles_dir=/workspace/testing3/puzzles:zero_num_games_per_iteration=20:learner_training_step=0"
    then
        echo "FAILED: $d"
    fi
    # python3 tools/rule_curriculum_metrics.py \
    #     --sgf $d/sgf/1.sgf \
    #     --manifest $d/testing3_full.tsv \
    #     --training-log $d/Training.log \
    #     --op-log $d/op.log \
    #     --iteration ${d##*iter} \
    #     --output $d/metrics.json

done
