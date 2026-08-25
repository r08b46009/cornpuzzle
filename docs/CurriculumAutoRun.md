# Automatic curriculum experiment

The automatic driver performs a fresh, resumable experiment without touching
an existing pilot folder:

1. create `weight_iter_0` in a new run directory;
2. evaluate it on the fixed 80 full puzzles;
3. select 20 distinct-puzzle curriculum tasks;
4. run 2,000 self-play games at 50 MCTS and 500 learner updates;
5. evaluate the new checkpoint on the same fixed validation set;
6. update the contextual-bandit Teacher with validation improvement;
7. select the next 20 tasks and repeat.

Run ten iterations from the repository root:

```bash
RUN_NAME=Teacher_auto_seed0 \
CURRICULUM_DIR=/workspace/curriculum_Teacher_auto_seed0 \
END_ITERATION=10 \
GPU=0123 \
tools/run_curriculum_loop.sh
```

To extend the same experiment later, run the same command with a larger
`END_ITERATION`, such as 100. Completed training, validation, Teacher updates,
and task selections are detected and skipped. Do not change `RUN_NAME` or
`CURRICULUM_DIR` when resuming.

Important outputs:

- models: `Teacher_auto_seed0/model/weight_iter_*.pt`
- Teacher state: `curriculum_Teacher_auto_seed0/teacher_state.json`
- per-round task manifests and metrics: `curriculum_Teacher_auto_seed0/rounds/`
- complete driver log: `curriculum_Teacher_auto_seed0/automatic_loop.log`
- isolated validation runs: `Validation_Teacher_auto_seed0_step*/`

The default validation set contains all 80 original full puzzles, uses 800
games, 50 MCTS, deterministic action selection, and no learner updates.
