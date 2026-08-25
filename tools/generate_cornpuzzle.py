#!/usr/bin/env python3

"""Generate a CornPuzzle txt file.

The current CornPuzzle loader accepts a plain text file with optional header
comments and blank-line separated 1/0 grids. This script generates a tiling
for a 7x14 board by default.

Pieces are generated as irregular connected polyominoes instead of rectangles.
The shapes come from a randomized spanning tree over the board, which keeps the
output valid while still producing non-trivial piece outlines.
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Set, Tuple


Cell = Tuple[int, int]


def grid_neighbors(cell: Cell, rows: int, cols: int) -> Iterable[Cell]:
    row, col = cell
    if row > 0:
        yield row - 1, col
    if row + 1 < rows:
        yield row + 1, col
    if col > 0:
        yield row, col - 1
    if col + 1 < cols:
        yield row, col + 1


def build_spanning_tree(rows: int, cols: int, rng: random.Random) -> Dict[Cell, Set[Cell]]:
    """Create a randomized spanning tree over the board cells."""

    start = (rng.randrange(rows), rng.randrange(cols))
    tree: Dict[Cell, Set[Cell]] = {start: set()}
    visited = {start}
    stack = [start]

    while stack:
        cell = stack[-1]
        candidates = [neighbor for neighbor in grid_neighbors(cell, rows, cols) if neighbor not in visited]
        if not candidates:
            stack.pop()
            continue

        next_cell = rng.choice(candidates)
        visited.add(next_cell)
        tree.setdefault(cell, set()).add(next_cell)
        tree.setdefault(next_cell, set()).add(cell)
        stack.append(next_cell)

    return tree


def collect_component(start: Cell, tree: Dict[Cell, Set[Cell]], remaining: Set[Cell]) -> Set[Cell]:
    """Return the connected component of *start* inside *remaining*."""

    component = {start}
    queue: Deque[Cell] = deque([start])

    while queue:
        cell = queue.popleft()
        for neighbor in tree[cell]:
            if neighbor in remaining and neighbor not in component:
                component.add(neighbor)
                queue.append(neighbor)

    return component


def find_components(tree: Dict[Cell, Set[Cell]], remaining: Set[Cell]) -> List[Set[Cell]]:
    """Split the remaining cells into tree components."""

    components: List[Set[Cell]] = []
    seen: Set[Cell] = set()

    for cell in remaining:
        if cell in seen:
            continue
        component = collect_component(cell, tree, remaining)
        components.append(component)
        seen.update(component)

    return components


def choose_piece_size(component_size: int, rng: random.Random, min_piece_size: int, max_piece_size: int) -> int:
    """Choose a random size for the next piece."""

    upper = min(component_size, max_piece_size)
    lower = min(min_piece_size, upper)
    if upper <= 1:
        return 1
    if lower < 1:
        lower = 1

    if lower == upper:
        return upper

    bias = rng.random() ** 0.5
    size = lower + int(bias * (upper - lower))
    return max(lower, min(size, upper))


def grow_piece(component: Set[Cell], tree: Dict[Cell, Set[Cell]], rng: random.Random, size: int) -> Set[Cell]:
    """Grow a connected subset of *component* with exactly *size* cells."""

    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(component):
        raise ValueError("size cannot exceed component size")

    root = rng.choice(tuple(component))
    piece = {root}
    frontier = {neighbor for neighbor in tree[root] if neighbor in component}

    while len(piece) < size:
        if not frontier:
            raise RuntimeError("unable to grow a piece of the requested size")

        next_cell = rng.choice(tuple(frontier))
        frontier.remove(next_cell)
        piece.add(next_cell)

        for neighbor in tree[next_cell]:
            if neighbor in component and neighbor not in piece:
                frontier.add(neighbor)

    return piece


def piece_to_grid(piece: Set[Cell]) -> List[str]:
    """Rasterize a piece into its minimal bounding-box grid."""

    min_row = min(row for row, _ in piece)
    max_row = max(row for row, _ in piece)
    min_col = min(col for _, col in piece)
    max_col = max(col for _, col in piece)

    grid = []
    for row in range(min_row, max_row + 1):
        line = []
        for col in range(min_col, max_col + 1):
            line.append("1" if (row, col) in piece else "0")
        grid.append("".join(line))

    return grid


def build_pieces(rows: int, cols: int, rng: random.Random, min_piece_size: int, max_piece_size: int) -> List[List[str]]:
    """Create an irregular tiling for the requested board size."""

    tree = build_spanning_tree(rows, cols, rng)
    remaining: Set[Cell] = {(row, col) for row in range(rows) for col in range(cols)}
    pieces: List[List[str]] = []

    while remaining:
        components = find_components(tree, remaining)
        component = rng.choice(components)
        size = choose_piece_size(len(component), rng, min_piece_size, max_piece_size)
        piece_cells = grow_piece(component, tree, rng, size)
        pieces.append(piece_to_grid(piece_cells))
        remaining.difference_update(piece_cells)

    rng.shuffle(pieces)
    return pieces


def render_puzzle(rows: int, cols: int, pieces: List[List[str]], seed: int) -> str:
    lines = [
        f"# generated_by generate_cornpuzzle.py",
        f"# seed {seed}",
        f"# rows {rows}",
        f"# cols {cols}",
        f"# active_rows {rows}",
        f"# active_cols {cols}",
        "",
    ]

    for index, piece in enumerate(pieces):
        if index > 0:
            lines.append("")
        lines.extend(piece)

    lines.append("")
    return "\n".join(lines)


def validate_pieces(rows: int, cols: int, pieces: List[List[str]]) -> None:
    total_area = 0
    for piece in pieces:
        height = len(piece)
        width = len(piece[0]) if piece else 0
        if height == 0 or width == 0:
            raise ValueError("empty piece generated")
        if any(len(row) != width for row in piece):
            raise ValueError("piece rows have inconsistent widths")
        if any(ch not in {"0", "1"} for row in piece for ch in row):
            raise ValueError("piece grids must contain only 0/1 cells")

        cells = {(r, c) for r, row in enumerate(piece) for c, ch in enumerate(row) if ch == "1"}
        if not cells:
            raise ValueError("empty piece generated")

        queue: Deque[Cell] = deque([next(iter(cells))])
        seen = set(queue)
        while queue:
            row, col = queue.popleft()
            for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= next_row < height and 0 <= next_col < width and piece[next_row][next_col] == "1":
                    cell = (next_row, next_col)
                    if cell not in seen:
                        seen.add(cell)
                        queue.append(cell)

        if len(seen) != len(cells):
            raise ValueError("piece is disconnected")

        total_area += len(cells)

    if total_area != rows * cols:
        raise ValueError(f"generated area {total_area} does not match board area {rows * cols}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a CornPuzzle txt file.")
    parser.add_argument("-o", "--output", type=Path, help="Output txt file. If omitted, print to stdout.")
    parser.add_argument("--output-dir", type=Path, help="Directory for batch output. Required when --count > 1.")
    parser.add_argument("--count", type=int, default=1, help="Number of puzzle files to generate. Default: 1")
    parser.add_argument(
        "--name-template",
        type=str,
        default="family_{index:03d}_L4_{rows}x{cols}.txt",
        help="Filename template for batch output. Available fields: index, rows, cols, seed.",
    )
    parser.add_argument("--rows", type=int, default=7, help="Puzzle rows. Default: 7")
    parser.add_argument("--cols", type=int, default=14, help="Puzzle cols. Default: 14")
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: 0")
    parser.add_argument("--min-piece-size", type=int, default=2, help="Minimum piece area when possible. Default: 2")
    parser.add_argument("--max-piece-size", type=int, default=12, help="Maximum piece area. Default: 12")
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise SystemExit("rows and cols must be positive")
    if args.rows > 7 or args.cols > 14:
        raise SystemExit("this generator is intended for boards up to 7x14 in the current CornPuzzle environment")
    if args.min_piece_size <= 0 or args.max_piece_size <= 0:
        raise SystemExit("min-piece-size and max-piece-size must be positive")
    if args.min_piece_size > args.max_piece_size:
        raise SystemExit("min-piece-size cannot exceed max-piece-size")
    if args.max_piece_size > args.rows * args.cols:
        raise SystemExit("max-piece-size cannot exceed board area")
    if args.count <= 0:
        raise SystemExit("count must be positive")
    if args.count > 1 and args.output_dir is None:
        raise SystemExit("--output-dir is required when --count > 1")
    if args.count == 1 and args.output is None and args.output_dir is not None:
        raise SystemExit("use --output for a single file, or combine --output-dir with --count > 1")

    if args.count == 1:
        rng = random.Random(args.seed)
        pieces = build_pieces(args.rows, args.cols, rng, args.min_piece_size, args.max_piece_size)
        validate_pieces(args.rows, args.cols, pieces)
        content = render_puzzle(args.rows, args.cols, pieces, args.seed)

        if args.output is None:
            print(content, end="")
        else:
            args.output.write_text(content, encoding="utf-8")
            print(f"Wrote {args.output}")
        return

    output_dir = args.output_dir
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, args.count + 1):
        seed = args.seed + index - 1
        rng = random.Random(seed)
        pieces = build_pieces(args.rows, args.cols, rng, args.min_piece_size, args.max_piece_size)
        validate_pieces(args.rows, args.cols, pieces)
        content = render_puzzle(args.rows, args.cols, pieces, seed)
        file_name = args.name_template.format(index=index, rows=args.rows, cols=args.cols, seed=seed)
        output_path = output_dir / file_name
        output_path.write_text(content, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()