# Two-Stage First Placement + Right-Trim Continuation

This version keeps the original fixed two-stage action space:

1. `SELECT_PIECE`: candidate shape + 0/180 rotation.
2. `SELECT_POSITION`: one of the existing board-coordinate action IDs.

No policy-head size or input-channel count is added, so existing two-stage
checkpoints remain tensor-shape compatible.

## Layer rule

A layer is the current **topmost unfinished row**.

- The first piece of a layer keeps a free position decision, but its position
  action must lie on the current layer row. The column is chosen by the agent.
- After that first piece, later pieces in the same layer are deterministic in
  position. The environment scans to the right (with active-column wrap) from
  the previous anchor and finds the first empty cell in the current layer row.
- The selected piece is already normalized/trimmed by the CornPuzzle shape
  utilities. Its first normalized occupied cell is anchored to that forced
  target. Therefore the second-stage position action still exists, but only
  that one coordinate is legal.
- When every active column in the current layer row is occupied, the layer is
  complete. The environment advances to the next topmost unfinished row and
  the next piece gets a new free first placement.

## State visibility

The continuation target is extra environment state. To keep the old network
shape unchanged, channel 0 (empty-cell channel) marks the forced target with
`0.5` instead of the normal empty value `1.0`. All other channel dimensions are
unchanged.

## Curriculum prefixes

If a curriculum prefix starts on a partially occupied topmost unfinished row,
the environment treats it as a continuation state and resumes at the leftmost
remaining empty cell. A completely empty current row starts with a free first
placement.

## Evaluation switches

`tools/run_rule_curriculum_loop.sh` also honors:

```bash
RUN_FULL80_EVAL=0
RUN_TESTING_EVAL=0
```

The curriculum mastery validation is still retained because it controls stage
advancement.
