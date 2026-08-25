# Strict 100% rule curriculum

## Two-stage CornPuzzle actions

This version removes the forced `firstEmpty` placement without changing the
network's fixed policy size (128 logits). One physical placement is encoded as
two environment actions:

1. `SELECT_PIECE`: `action_id = candidate_id * 2 + rotation_id`.
2. `SELECT_POSITION`: `action_id = row * 14 + col` (`0..97`).

The piece-selection action has zero immediate reward. The position action
places the pending rotated shape, applies active-column wrapping, and receives
the original progress/completion reward. Illegal coordinates are masked.

The input tensor remains 67 x 14 x 14. During `SELECT_POSITION`, all candidate
planes except the pending piece are zero; the pending piece's rotated geometry
is encoded as `-1`. This identifies the phase, piece, and rotation without
changing the embedding-convolution shape.

Compatibility notes:

- Start a new training directory. Old SGFs contain one action per placement
  and cannot be mixed with the new two-action trajectories.
- Existing weights have compatible tensor dimensions and may be used only as
  initialization. Position-action semantics are new and must be learned.
- Game lengths are approximately doubled. A 22-piece solution now has 44
  environment actions.
- Numeric SGF action IDs remain replayable because phase is reconstructed by
  alternating piece and position states.

This experiment trains a fresh AlphaZero Student without a learned Teacher.
It uses a deterministic sliding curriculum:

1. `remain2`
2. `remain2 + remain4`
3. `remain4 + remain6`
4. Continue in steps of two through `remain20 + remain22`
5. `remain22 + full`
6. `full`

## Advancement certificate

A stage advances only when **both consecutive validation rounds** satisfy all
of the following for every active level:

- solve rate is exactly 100%;
- all 80 puzzle task types were sampled;
- every sampled game for every task was solved.

One failed game or one uncovered task resets the consecutive-pass counter.
The default 1,600 validation games give about 20 trials/task in a one-level
stage and 10 trials/task in a two-level stage.

## Start a new experiment

Run from the repository root inside the container:

```bash
chmod +x tools/run_rule_curriculum_loop.sh

RUN_NAME=RuleCurriculum_seed0 \
CURRICULUM_DIR=/workspace/curriculum_RuleCurriculum_seed0 \
PLOTS_DIR=/workspace/plots_RuleCurriculum_seed0 \
END_ITERATION=200 \
GPU=0123 \
ZERO_SERVER_PORT=22340 \
TESTING_PUZZLES_DIR=/workspace/validation \
TESTING_GAMES=200 \
tools/run_rule_curriculum_loop.sh
```

Use a new `RUN_NAME`, `CURRICULUM_DIR`, and port if another experiment is
running. Reusing those names resumes the same run safely.

## Outputs

- Student: `RuleCurriculum_seed0/`
- Rule state and per-round metrics: `curriculum_RuleCurriculum_seed0/`
- Continuously refreshed charts: `plots_RuleCurriculum_seed0/`

The charts contain shaded stage regions and transition lines:

- policy accuracy;
- policy loss;
- value loss;
- min/average/max returns;
- min/average/max game lengths;
- self-play, optimization, and total time;
- held-out testing accuracy read directly from each
  `TestingRule_<run>_iter<N>/sgf/1.sgf` file.

The testing set is never used for training or curriculum promotion. The
default 200 held-out games after each iteration add real MCTS work—roughly 10%
of the 2,000 training-game search count.

CornPuzzle color output follows the existing
`program_use_color_message=true` cfg setting. `CORNPUZZLE_COLOR` remains an
optional override, but no terminal export is required. Because this changes
C++, rebuild once before the first run.

Training uses 50 MCTS simulations throughout and a replay buffer of four
iterations. The short replay window lets the previous/easier level fade after
the curriculum window moves forward.
