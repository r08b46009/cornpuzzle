#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 CONFIG_FILE [CONFIG_FILE_STR] MODEL_PT_FILE EVAL_PUZZLES_DIR EVAL_RESULT_OUTPUT_DIR"
    echo ""
    echo "Run MiniZero CornPuzzle evaluation, then convert stdout results into HTML and final-board PNG files."
    echo ""
    echo "Arguments:"
    echo "  CONFIG_FILE              MiniZero config file path"
    echo "  CONFIG_FILE_STR          optional colon-separated config overrides"
    echo "  MODEL_PT_FILE            model .pt file path"
    echo "  EVAL_PUZZLES_DIR         directory containing evaluation puzzle .txt files"
    echo "  EVAL_RESULT_OUTPUT_DIR   output directory for stdout, sgf, html, and png files"
    echo ""
    echo "Example:"
    echo "  $0 0816_plateau_au_2/0816_plateau_au_2.cfg \\"
    echo "     0816_plateau_au_2/model/weight_iter_90000.pt \\"
    echo "     data/generated_wrap/puzzles/71424 \\"
    echo "     solve_outputs/run_001"
    echo ""
    echo "Example with config overrides:"
    echo "  $0 0816_plateau_au_2/0816_plateau_au_2.cfg \\"
    echo "     \"actor_num_simulation=400:actor_select_action_by_count=true:actor_select_action_by_softmax_count=false\" \\"
    echo "     0816_plateau_au_2/model/weight_iter_90000.pt \\"
    echo "     data/generated_wrap/puzzles/71424 \\"
    echo "     solve_outputs/run_001"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 4 && $# -ne 5 ]]; then
    usage >&2
    exit 1
fi

config_file=$1
if [[ $# -eq 5 ]]; then
    config_file_str=$2
    model_file=$3
    puzzles_dir=$4
    output_dir=$5
else
    config_file_str=""
    model_file=$2
    puzzles_dir=$3
    output_dir=$4
fi

executable="./build/cornpuzzle/minizero_cornpuzzle"
stdout_dir="${output_dir}/stdout"
final_config_file="${output_dir}/eval_config.cfg"

if [[ ! -x ${executable} ]]; then
    echo "MiniZero executable not found or not executable: ${executable}" >&2
    echo "Build first with: ./scripts/build.sh cornpuzzle release" >&2
    exit 1
fi

conf_str="nn_file_name=${model_file}:env_compound_puzzles_dir=${puzzles_dir}:test_output_path=${output_dir}"
if [[ -n ${config_file_str} ]]; then
    conf_str="${conf_str}:${config_file_str}"
fi

mkdir -p "${output_dir}"
tmp_config_file="${output_dir}/.eval_config.$$"

echo "[run_cornpuzzle_eval_plot] Writing final evaluation config"
"${executable}" \
    -conf_file "${config_file}" \
    -conf_str "${conf_str}" \
    -gen "${tmp_config_file}" >/dev/null
mv "${tmp_config_file}" "${final_config_file}"

echo "[run_cornpuzzle_eval_plot] Running MiniZero evaluation"
"${executable}" \
    -mode solve_cornpuzzle \
    -conf_file "${config_file}" \
    -conf_str "${conf_str}"

echo "[run_cornpuzzle_eval_plot] Rendering HTML and PNG outputs"
python3 tools/solve_stdout_to_html.py "${stdout_dir}"

echo "[run_cornpuzzle_eval_plot] Done"
echo "  config: ${final_config_file}"
echo "  stdout: ${stdout_dir}"
echo "  sgf:    ${output_dir}/sgf"
echo "  html:   ${output_dir}/html"
echo "  png:    ${output_dir}/html"
