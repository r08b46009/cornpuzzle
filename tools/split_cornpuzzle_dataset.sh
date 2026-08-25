#!/bin/bash
# Randomly split a folder of CornPuzzle puzzle files into a train/test set.
# Files are COPIED -- SRC_DIR is left untouched.
set -euo pipefail

usage() {
	cat <<EOF
Usage: $0 [SRC_DIR] [TRAIN_DIR] [TEST_DIR] [TEST_COUNT] [SEED]

Randomly split the puzzle files in SRC_DIR into TRAIN_DIR and TEST_DIR.
Files are copied; SRC_DIR itself is never modified or deleted from.

Defaults:
  SRC_DIR:    cornpuzzle/71424
  TRAIN_DIR:  cornpuzzle/71424_train
  TEST_DIR:   cornpuzzle/71424_test
  TEST_COUNT: 20      (remaining files go to TRAIN_DIR)
  SEED:       42      (fixed seed -> same split every time you run this)

Example:
  $0 cornpuzzle/71424 cornpuzzle/71424_train cornpuzzle/71424_test 20 42
EOF
	exit 1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
	usage
fi

src_dir=${1:-cornpuzzle/71424}
train_dir=${2:-cornpuzzle/71424_train}
test_dir=${3:-cornpuzzle/71424_test}
test_count=${4:-20}
seed=${5:-42}

if [ ! -d "$src_dir" ]; then
	echo "Error: source folder not found: $src_dir" >&2
	exit 1
fi

mapfile -t files < <(find "$src_dir" -maxdepth 1 -type f | sort)
total=${#files[@]}

if [ "$total" -eq 0 ]; then
	echo "Error: no files found in $src_dir" >&2
	exit 1
fi
if [ "$test_count" -ge "$total" ]; then
	echo "Error: TEST_COUNT ($test_count) must be less than the number of files found ($total)" >&2
	exit 1
fi

# Deterministic shuffle: pair each file with a seeded pseudo-random key, then
# sort by that key. Same SEED -> same split every run.
mapfile -t shuffled < <(
	printf '%s\n' "${files[@]}" \
		| awk -v seed="$seed" 'BEGIN { srand(seed) } { print rand() "\t" $0 }' \
		| sort -n \
		| cut -f2-
)

test_files=("${shuffled[@]:0:test_count}")
train_files=("${shuffled[@]:test_count}")

mkdir -p "$train_dir" "$test_dir"

for f in "${test_files[@]}"; do
	cp "$f" "$test_dir/"
done
for f in "${train_files[@]}"; do
	cp "$f" "$train_dir/"
done

echo "Source: $src_dir ($total files)"
echo "Test:   $test_dir (${#test_files[@]} files, seed=$seed)"
echo "Train:  $train_dir (${#train_files[@]} files)"
