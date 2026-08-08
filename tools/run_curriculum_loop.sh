#!/bin/bash
# Automatic CornPuzzle curriculum experiment. Safe to resume after interruption.
set -Eeuo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "${project_dir}"

run_name="${RUN_NAME:-Teacher_auto_seed0}"
curriculum_dir="${CURRICULUM_DIR:-${project_dir}/curriculum_${run_name}}"
end_iteration="${END_ITERATION:-10}"
start_iteration="${START_ITERATION:-1}"
gpu="${GPU:-0123}"
base_config="${BASE_CONFIG:-configs/CL_new_wo_RT.cfg}"
puzzles_dir="${PUZZLES_DIR:-${project_dir}/data/generated_wrap/puzzles/71424}"
answers_dir="${ANSWERS_DIR:-${project_dir}/data/generated_wrap/answers}"
validation_games="${VALIDATION_GAMES:-800}"
total_iterations="${TOTAL_ITERATIONS:-100}"
training_timeout="${TRAINING_TIMEOUT:-15m}"
validation_timeout="${VALIDATION_TIMEOUT:-10m}"
max_retries="${MAX_RETRIES:-3}"
validation_prefix="${VALIDATION_PREFIX:-ValidationCompletion}"

task_bank="${curriculum_dir}/task_bank.json"
validation_tasks="${curriculum_dir}/validation_endgame_diagnostic.tsv"
active_tasks="${curriculum_dir}/active_tasks.tsv"
teacher_state="${curriculum_dir}/teacher_state.json"
rounds_dir="${curriculum_dir}/rounds"
master_log="${curriculum_dir}/automatic_loop.log"

mkdir -p "${rounds_dir}"
exec > >(tee -a "${master_log}") 2>&1

fail() { echo "[curriculum][ERROR] $*" >&2; exit 1; }
weight_step() { echo $(( $1 * 500 )); }

run_with_retry() {
    local label=$1 limit=$2 attempt status
    shift 2
    for ((attempt=1; attempt<=max_retries; attempt++)); do
        echo "[curriculum] ${label}: attempt ${attempt}/${max_retries}, timeout=${limit}"
        set +e
        timeout --signal=INT --kill-after=90s "${limit}" "$@"
        status=$?
        set -e
        if (( status == 0 )); then
            return 0
        fi
        echo "[curriculum][WARN] ${label} failed/timed out (status=${status})"
        sleep 10
    done
    fail "${label} failed after ${max_retries} attempts"
}

[[ -f "${base_config}" ]] || fail "missing config: ${base_config}"
[[ -d "${puzzles_dir}" ]] || fail "missing puzzles: ${puzzles_dir}"
[[ -d "${answers_dir}" ]] || fail "missing answers: ${answers_dir}"

if [[ ! -f "${task_bank}" ]]; then
    python3 tools/curriculum_teacher.py build \
        --puzzles "${puzzles_dir}" --answers "${answers_dir}" \
        --output "${task_bank}" --expected-puzzles 80
fi
if [[ ! -f "${validation_tasks}" ]]; then
    python3 tools/curriculum_teacher.py validation \
        --task-bank "${task_bank}" --output "${validation_tasks}"
fi

run_validation() {
    local iteration=$1 step vdir metrics sgf
    step=$(weight_step "${iteration}")
    vdir="${validation_prefix}_${run_name}_step${step}"
    metrics="${rounds_dir}/after_iter${iteration}.json"
    sgf="${vdir}/sgf/1.sgf"

    if [[ -f "${metrics}" && -f "${vdir}/.validation_complete" ]]; then
        echo "[curriculum] validation ${iteration} already complete"
        return
    fi
    [[ -f "${run_name}/model/weight_iter_${step}.pt" ]] || fail "missing weight_iter_${step}.pt"
    [[ -f "${run_name}/model/weight_iter_${step}.pkl" ]] || fail "missing weight_iter_${step}.pkl"

    mkdir -p "${vdir}/model" "${vdir}/sgf"
    cp "${run_name}/${run_name}.cfg" "${vdir}/${vdir}.cfg"
    cp "${run_name}/model/weight_iter_${step}.pt" "${vdir}/model/"
    cp "${run_name}/model/weight_iter_${step}.pkl" "${vdir}/model/"
    touch "${vdir}/op.log"

    run_with_retry "validation iteration ${iteration}" "${validation_timeout}" \
    env MINIZERO_RUN_STAGE=C MINIZERO_CONFIRM_CONTINUE=y \
    tools/quick-run.sh train cornpuzzle "${vdir}/${vdir}.cfg" 1 \
        -n "${vdir}" -g "${gpu}" --sp_progress \
        -conf_str "actor_num_simulation=50:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false:actor_use_dirichlet_noise=false:actor_use_gumbel_noise=false:env_cornpuzzle_curriculum_enable=true:env_cornpuzzle_curriculum_tasks_file=${validation_tasks}:zero_num_games_per_iteration=${validation_games}:learner_training_step=0"

    [[ -f "${sgf}" ]] || fail "validation SGF missing: ${sgf}"
    python3 tools/curriculum_metrics.py \
        --training-log "${vdir}/Training.log" --op-log "${vdir}/op.log" \
        --sgf "${sgf}" --iteration-progress "$(awk -v i="${iteration}" -v n="${total_iterations}" 'BEGIN { print i/n }')" \
        --output "${metrics}"
    touch "${vdir}/.validation_complete"
}

# A brand-new formal run starts with weight_iter_0 and its fixed diagnostic
# validation (full, remain2, remain4, remain6, and remain10 tasks).
if [[ ! -d "${run_name}" ]]; then
    echo "[curriculum] creating fresh run: ${run_name}"
    # END_ITERATION=0 asks zero-server to create the configuration and random
    # weight_iter_0 without producing self-play data. Some MiniZero revisions
    # return a non-zero status after this initialization-only run, so the
    # checkpoint below is the authoritative success condition.
    MINIZERO_RUN_STAGE=R tools/quick-run.sh train cornpuzzle "${base_config}" 0 \
        -n "${run_name}" -g "${gpu}" \
        -conf_str "program_seed=0:program_auto_seed=false:actor_num_simulation=50:env_compound_puzzles_dir=${puzzles_dir}" || true
fi
[[ -f "${run_name}/model/weight_iter_0.pt" ]] || fail "fresh weight_iter_0.pt was not created"

if (( start_iteration < 1 || start_iteration > end_iteration )); then
    fail "START_ITERATION must be between 1 and END_ITERATION"
fi

# For a fresh run, bootstrap from weight 0. To attach a new Teacher to an
# existing Student, bootstrap from the checkpoint immediately before
# START_ITERATION (e.g. START_ITERATION=18 uses weight_iter_8500).
bootstrap_iteration=$((start_iteration - 1))
run_validation "${bootstrap_iteration}"

if [[ ! -f "${active_tasks}" ]]; then
    python3 tools/curriculum_teacher.py select \
        --task-bank "${task_bank}" --metrics "${rounds_dir}/after_iter${bootstrap_iteration}.json" \
        --state "${teacher_state}" --output "${active_tasks}" --seed "${bootstrap_iteration}"
    cp "${active_tasks}" "${rounds_dir}/iter${start_iteration}_tasks.tsv"
fi

for ((iteration=start_iteration; iteration<=end_iteration; iteration++)); do
    step=$(weight_step "${iteration}")
    echo "[curriculum] ===== iteration ${iteration}/${end_iteration}; target step ${step} ====="

    if [[ ! -f "${run_name}/model/weight_iter_${step}.pt" ]]; then
        run_with_retry "training iteration ${iteration}" "${training_timeout}" \
        env MINIZERO_RUN_STAGE=C MINIZERO_CONFIRM_CONTINUE=y \
        tools/quick-run.sh train cornpuzzle "${base_config}" "${iteration}" \
            -n "${run_name}" -g "${gpu}" --sp_progress \
            -conf_str "program_seed=0:program_auto_seed=false:actor_num_simulation=50:env_cornpuzzle_curriculum_enable=true:env_cornpuzzle_curriculum_tasks_file=${active_tasks}:env_compound_puzzles_dir=${puzzles_dir}"
    else
        echo "[curriculum] weight_iter_${step}.pt already exists; skip training"
    fi

    run_validation "${iteration}"

    # Add this round's training-only measurements to the validation metrics.
    python3 - "${rounds_dir}/after_iter${iteration}.json" "${run_name}/Training.log" "${run_name}/op.log" "${iteration}" <<'PY'
import json, re, sys
from pathlib import Path
p = Path(sys.argv[1]); d = json.loads(p.read_text())
training = Path(sys.argv[2]).read_text(errors="replace")
op = Path(sys.argv[3]).read_text(errors="replace")
iteration = int(sys.argv[4])
def last(pattern, text, default=0.0):
    x = re.findall(pattern, text, re.M | re.S); return float(x[-1]) if x else default

# Select the requested iteration block, rather than the last block in the log.
blocks = re.findall(
    rf"\[Iteration\]\s*=+{iteration}=+(.*?)(?=\[Iteration\]\s*=+\d+=+|\Z)",
    training,
    re.S,
)
# Retries can create more than one block for the same iteration.  The last
# block is the completed retry that produced the checkpoint.
block = blocks[-1] if blocks else ""
d["train_return"] = last(r"\[SelfPlay Avg\. Game Returns\]\s+([-+0-9.eE]+)", block)

# Learner metrics are keyed by the exact cumulative target step.
target_step = iteration * 500
step_match = re.search(
    rf"nn step {target_step},.*?(?=nn step \d+,|\[command\]|\Z)",
    op,
    re.S,
)
step_block = step_match.group(0) if step_match else ""
d["policy_loss"] = last(r"loss_policy:\s*([-+0-9.eE]+)", step_block)
d["policy_accuracy"] = last(r"accuracy_policy:\s*([-+0-9.eE]+)", step_block)
d["value_loss"] = last(r"loss_value:\s*([-+0-9.eE]+)", step_block)
p.write_text(json.dumps(d, indent=2) + "\n")
PY

    if [[ ! -f "${rounds_dir}/iter${iteration}_teacher_updated" ]]; then
        python3 tools/curriculum_teacher.py update \
            --state "${teacher_state}" --metrics "${rounds_dir}/after_iter${iteration}.json"
        touch "${rounds_dir}/iter${iteration}_teacher_updated"
    fi

    if (( iteration < end_iteration )); then
        next=$((iteration + 1))
        if [[ ! -f "${rounds_dir}/iter${next}_tasks.tsv" ]]; then
            python3 tools/curriculum_teacher.py select \
                --task-bank "${task_bank}" --metrics "${rounds_dir}/after_iter${iteration}.json" \
                --state "${teacher_state}" --output "${active_tasks}" --seed "${iteration}"
            cp "${active_tasks}" "${rounds_dir}/iter${next}_tasks.tsv"
        else
            cp "${rounds_dir}/iter${next}_tasks.tsv" "${active_tasks}"
        fi
    fi
done

echo "[curriculum] completed through iteration ${end_iteration}"
echo "[curriculum] latest model: ${run_name}/model/weight_iter_$(weight_step "${end_iteration}").pt"
