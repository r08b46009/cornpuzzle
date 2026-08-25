# Completion-driven Teacher curriculum (v2)

This implementation does **not** generate an 81st puzzle and does not use CKL/200-search refinement.
The Student always uses 50 MCTS. A task is one of the original 80 puzzles plus a difficulty level:
`full`, followed by answer-derived states every two remaining pieces (`remain22`, `remain20`, ...,
`remain4`, `remain2`, depending on that puzzle's piece count). Twenty distinct original puzzles are
selected per round, with one difficulty level chosen for each puzzle. These 20 task *types* are
per round and are sampled repeatedly until the normal self-play game budget is reached.

The candidate pool is fixed after `build`, but the selected 20 and their scores change every round.
For 24-piece puzzles this is 12 levels per puzzle, or about 960 task types for 80 puzzles.

The Teacher is a contextual LinUCB bandit. It never learns how to solve a puzzle. It selects exactly
20 guaranteed-solvable answer-prefix tasks. The allocation starts endgame-heavy and automatically
moves toward full-board practice:

| Band | Initial tasks | Purpose |
| --- | ---: | --- |
| `remain2` | 4 | final compatibility decision |
| `remain4` | 4 | short endgame planning |
| `remain6` | 4 | avoid sealing required space |
| `remain8`--`remain12` | 4 | bridge endgame to midgame |
| `full` | 4 | prevent forgetting the real task |

Within each band, three tasks use the LinUCB score and one is a least-seen exploration task. Only
one difficulty level from any original puzzle may appear in a round.

The stage schedule is:

| Stage | remain2 | remain4 | remain6 | remain10 | full |
| --- | ---: | ---: | ---: | ---: | ---: |
| full20 | 4 | 4 | 4 | 4 | 4 |
| full40 | 3 | 3 | 2 | 4 | 8 |
| full60 | 2 | 1 | 1 | 4 | 12 |
| full80 | 1 | 1 | 0 | 2 | 16 |

The Teacher enters `full40` when endgame solve rate reaches 70%, `full60` when endgame reaches 85%
and remain10 reaches 60%, and `full80` when full solve rate reaches 50%.

The fixed diagnostic validation manifest contains `full`, `remain2`, `remain4`, `remain6`, and
`remain10` states. For full failures, completion shaping is:

```text
completion_score = solve_rate
                 + 0.25 * P(fail one piece short)
                 + 0.10 * P(fail two pieces short)
                 + 0.05 * P(fail three pieces short)
```

The Teacher reward is:

```text
0.50 * delta(full solve rate)
+ 0.25 * delta(completion score)
+ 0.15 * delta(mean remain2/4/6 solve rate)
+ 0.10 * delta(full average return)
```

This gives a useful signal while full solve rate is still zero, but a true solve remains worth much
more than merely ending one piece short. Within a difficulty band, task ranking combines LinUCB,
closeness to a 50% learning frontier, and uncertainty; one slot remains least-seen exploration.

## One-time task bank

```bash
PROJECT="$(pwd)"
tools/prepare_curriculum.sh "${PROJECT}/curriculum"
```

## Recommended fresh automatic run

Do not reuse an old `teacher_state.json`. Start from a new run and curriculum directory:

```bash
cd /workspace

RUN_NAME=Teacher_completion_seed0 \
CURRICULUM_DIR=/workspace/curriculum_Teacher_completion_seed0 \
END_ITERATION=100 \
GPU=0123 \
VALIDATION_GAMES=800 \
tools/run_curriculum_loop.sh
```

With 800 diagnostic games and 400 fixed task types, each of the five diagnostic bands receives
about 160 games in expectation. Increase `VALIDATION_GAMES` to 1600 for smoother Teacher rewards
if the additional evaluation cost is acceptable.

### Attach v2 Teacher to an existing Student checkpoint

Student weights do not need to restart. If the latest checkpoint is `weight_iter_8500`, its completed
iteration is 17 and the next training iteration is 18:

```bash
cd /workspace

RUN_NAME=Teacher_auto_seed0 \
CURRICULUM_DIR=/workspace/curriculum_Teacher_completion_from8500 \
START_ITERATION=18 \
END_ITERATION=100 \
GPU=0123 \
VALIDATION_GAMES=800 \
tools/run_curriculum_loop.sh
```

Use a new `CURRICULUM_DIR` so the old return-only LinUCB state is not mixed with the new reward.
The script first evaluates `weight_iter_8500`, creates the new Teacher state, selects tasks for
iteration 18, and then continues Student training from that checkpoint.

## One curriculum round

Write the pre-training fixed-validation result to `before.json`, for example:

```json
{"iteration_progress": 0.1, "validation_return": 0.31, "validation_solve_rate": 0.08,
 "endgame_solve_rate": 0.72, "remain2_solve_rate": 0.95,
 "remain4_solve_rate": 0.75, "remain6_solve_rate": 0.46,
 "middle_solve_rate": 0.22, "near_finish_failure_rate": 0.18,
 "validation_avg_length": 18.2, "train_return": 0.29,
 "policy_loss": 2.1, "value_loss": 0.12}
```

Select exactly 20 task types, then train normally. The command reuses the clean
`CL_new_wo_RT.cfg`; only the curriculum switch, paths, and fixed 50-MCTS budget are overridden:

```bash
PROJECT="$(pwd)"
python3 tools/curriculum_teacher.py select \
  --task-bank "${PROJECT}/curriculum/task_bank.json" \
  --metrics "${PROJECT}/curriculum/before.json" \
  --state "${PROJECT}/curriculum/teacher_state.json" \
  --output "${PROJECT}/curriculum/active_tasks.tsv"

tools/quick-run.sh train cornpuzzle configs/CL_new_wo_RT.cfg 1 \
  -n Teacher_seed0 -g 0123 --sp_progress \
  -conf_str "actor_num_simulation=50:env_cornpuzzle_curriculum_enable=true:env_cornpuzzle_curriculum_tasks_file=${PROJECT}/curriculum/active_tasks.tsv:env_compound_puzzles_dir=${PROJECT}/data/generated_wrap/puzzles/71424"
```

Evaluate the new Student on the same fixed validation set, write `after.json`, and update:

```bash
python3 tools/curriculum_teacher.py update \
  --state "${PROJECT}/curriculum/teacher_state.json" \
  --metrics "${PROJECT}/curriculum/after.json"
```

Repeat select → Student training → fixed validation → update.

## Fair comparison

Use the same seed list, initial weights, network, optimizer, 50 MCTS, number of Student self-play games,
training steps, and fixed validation set for all groups:

1. **Baseline:** `env_cornpuzzle_curriculum_enable=false`, random sampling from all 80 full puzzles.
2. **Random-20 control:** run `select --strategy random`, curriculum enabled.
3. **Teacher-20:** default endgame-aware learned selection.

Use a **new** `RUN_NAME` and `CURRICULUM_DIR`. The context dimension and state format differ from
the earlier return-only Teacher, so its `teacher_state.json` must not be reused.

The answer grids are used only to construct starting states. They are not inserted as supervised policy
targets; learner samples contain only actions produced by the Student after the curriculum start state.
Invalid answer-derived states are rejected by the environment rather than trained on.

Baseline command using the same clean configuration:

```bash
PROJECT="$(pwd)"
tools/quick-run.sh train cornpuzzle configs/CL_new_wo_RT.cfg 1 \
  -n Baseline80_seed0 -g 0123 --sp_progress \
  -conf_str "actor_num_simulation=50:env_cornpuzzle_curriculum_enable=false:env_compound_puzzles_dir=${PROJECT}/data/generated_wrap/puzzles/71424:env_compound_random_select_puzzle=true"
```
