#!/bin/bash
set -e
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
curriculum_dir="${1:-${project_dir}/curriculum}"
mkdir -p "${curriculum_dir}"
python3 "${project_dir}/tools/curriculum_teacher.py" build \
  --puzzles "${project_dir}/data/generated_wrap/puzzles/71424" \
  --answers "${project_dir}/data/generated_wrap/answers" \
  --output "${curriculum_dir}/task_bank.json"
echo "Task bank ready: ${curriculum_dir}/task_bank.json"
