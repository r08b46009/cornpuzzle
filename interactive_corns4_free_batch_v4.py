import sys
import argparse
import os
import random
import re
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Iterable, Optional, Callable

COLOR_MAP = {
    1: "\033[91m",  # red
    2: "\033[92m",  # green
    3: "\033[93m",  # yellow
    4: "\033[94m",  # blue
    5: "\033[95m",  # magenta
    6: "\033[96m",  # cyan
    7: "\033[97m",  # white
}
RESET = "\033[0m"
SYMBOLS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ----------------------------
# Piece parsing and transforms
# ----------------------------

def read_pieces(puzzle_file: Optional[str] = None) -> List[List[List[List[int]]]]:
    """Read pieces separated by blank lines and group identical shapes."""
    pieces: List[List[List[int]]] = []
    current: List[List[int]] = []
    f = open(puzzle_file, "r", encoding="utf-8") if puzzle_file else sys.stdin
    try:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if current:
                    pieces.append(current)
                    current = []
                continue
            current.append([int(c) for c in line])
        if current:
            pieces.append(current)
    finally:
        if puzzle_file:
            f.close()

    groups: Dict[Tuple[Tuple[int, ...], ...], List[List[List[int]]]] = defaultdict(list)
    for piece in pieces:
        groups[tuple(tuple(row) for row in piece)].append(piece)
    return list(groups.values())


def rotate_90(piece: List[List[int]]) -> List[List[int]]:
    return [list(row) for row in zip(*piece[::-1])]


def count_area(piece: List[List[int]]) -> int:
    return sum(sum(row) for row in piece)


def symbol_for_group(group_id: int) -> str:
    if 0 < group_id <= len(SYMBOLS):
        return SYMBOLS[group_id - 1]
    return "?"


def piece_cells(piece: List[List[int]]) -> List[Tuple[int, int]]:
    cells = []
    for r, row in enumerate(piece):
        for c, val in enumerate(row):
            if val == 1:
                cells.append((r, c))
    return cells


def cells_to_piece(cells: Iterable[Tuple[int, int]]) -> List[List[int]]:
    cells = list(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = [(r - min_r, c - min_c) for r, c in cells]
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    grid = [[0 for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for r, c in norm:
        grid[r][c] = 1
    return grid


def generate_variants(piece: List[List[int]]) -> List[Tuple[int, Tuple[Tuple[int, int], ...]]]:
    """Return unique 0/180 rotations as (angle, normalized relative cell coords)."""
    seen = set()
    variants = []

    for angle in (0, 180):
        current = piece
        if angle == 180:
            current = rotate_90(rotate_90(current))

        cells = piece_cells(current)
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        rel_cells = tuple(sorted((r - min_r, c - min_c) for r, c in cells))

        if rel_cells not in seen:
            seen.add(rel_cells)
            variants.append((angle, rel_cells))

    return variants


def render_piece_from_cells(cells: Iterable[Tuple[int, int]]) -> List[str]:
    cells = list(cells)
    if not cells:
        return []
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    canvas = [["." for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for r, c in cells:
        canvas[r][c] = "#"
    return ["".join(row) for row in canvas]


# ----------------------------
# Solver
# ----------------------------

ProgressCallback = Callable[[int, int, int, int], None]


class ExactishSolver:
    """Backtracking solver with simple CSP heuristics for wrap-around polyomino tiling."""

    def __init__(self, rows: int, cols: int, wrap_cols: bool, group_defs: List[dict]):
        self.rows = rows
        self.cols = cols
        self.wrap_cols = wrap_cols
        self.group_defs = group_defs
        self.node_count = 0
        self.best_depth = 0
        self.first_solution_node = None
        self.first_solution_depth = None
        self.first_solution_filled = None

    def first_empty(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] == 0:
                    return (r, c)
        return None

    def area_matches_remaining(self, board: List[List[int]], remaining: List[int]) -> bool:
        empty = sum(1 for row in board for cell in row if cell == 0)
        need = sum(rem * self.group_defs[idx]["area"] for idx, rem in enumerate(remaining))
        return need == empty

    def can_place(
        self,
        board: List[List[int]],
        rel_cells: Tuple[Tuple[int, int], ...],
        top: int,
        left: int,
    ) -> Optional[List[Tuple[int, int]]]:
        changes = []
        for dr, dc in rel_cells:
            r = top + dr
            c = left + dc
            if self.wrap_cols:
                c %= self.cols
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return None
            if board[r][c] != 0:
                return None
            changes.append((r, c))
        return changes

    def place(self, board: List[List[int]], changes: List[Tuple[int, int]], mark: int) -> None:
        for r, c in changes:
            board[r][c] = mark

    def unplace(self, board: List[List[int]], changes: List[Tuple[int, int]]) -> None:
        for r, c in changes:
            board[r][c] = 0

    def placements_covering_target(
        self,
        board: List[List[int]],
        target: Tuple[int, int],
        group_idx: int,
    ) -> List[Tuple[int, List[Tuple[int, int]], Tuple[Tuple[int, int], ...], int, int]]:
        target_r, target_c = target
        candidates = []
        for angle, rel_cells in self.group_defs[group_idx]["variants"]:
            for cell_r, cell_c in rel_cells:
                top = target_r - cell_r
                left = target_c - cell_c
                changes = self.can_place(board, rel_cells, top, left)
                if changes is not None:
                    candidates.append((angle, changes, rel_cells, top, left))
        dedup = {}
        for angle, changes, rel_cells, top, left in candidates:
            key = tuple(sorted(changes))
            if key not in dedup:
                dedup[key] = (angle, changes, rel_cells, top, left)
        return list(dedup.values())

    def ordered_groups(self, board: List[List[int]], remaining: List[int], target: Tuple[int, int]):
        options = []
        for idx, rem in enumerate(remaining):
            if rem <= 0:
                continue
            cands = self.placements_covering_target(board, target, idx)
            if cands:
                options.append((idx, cands))
        options.sort(key=lambda item: len(item[1]))
        return options

    def solve_from_state(self, board: List[List[int]], remaining: List[int]):
        self.node_count = 0
        self.best_depth = 0
        self.first_solution_node = None
        self.first_solution_depth = None
        self.first_solution_filled = None
        working = [row[:] for row in board]
        success = self._dfs(working, remaining[:])
        return success, working, self.node_count

    def count_solutions_from_state(
        self,
        board: List[List[int]],
        remaining: List[int],
        max_solutions: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.node_count = 0
        self.best_depth = 0
        self.first_solution_node = None
        self.first_solution_depth = None
        self.first_solution_filled = None
        working = [row[:] for row in board]
        count = self._count_dfs(
            working,
            remaining[:],
            max_solutions=max_solutions,
            progress_callback=progress_callback,
        )
        return count, self.node_count

    def _placement_score(self, board: List[List[int]], changes: List[Tuple[int, int]]) -> int:
        score = 0
        change_set = set(changes)
        for r, c in changes:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nr < 0 or nr >= self.rows:
                    score += 1
                    continue
                if self.wrap_cols:
                    nc %= self.cols
                elif nc < 0 or nc >= self.cols:
                    score += 1
                    continue
                if board[nr][nc] == 0 and (nr, nc) not in change_set:
                    score += 1
        return score

    def _dfs(self, board: List[List[int]], remaining: List[int], depth: int = 0) -> bool:
        self.node_count += 1
        filled = sum(cell != 0 for row in board for cell in row)
        self.best_depth = max(self.best_depth, filled)

        target = self.first_empty(board)
        if target is None:
            solved = all(rem == 0 for rem in remaining)
            if solved and self.first_solution_node is None:
                self.first_solution_node = self.node_count
                self.first_solution_depth = depth
                self.first_solution_filled = filled
            return solved
        if not self.area_matches_remaining(board, remaining):
            return False

        group_options = self.ordered_groups(board, remaining, target)
        if not group_options:
            return False

        for group_idx, candidates in group_options:
            candidates.sort(key=lambda item: self._placement_score(board, item[1]))
            remaining[group_idx] -= 1
            mark = self.group_defs[group_idx]["id"]
            for _angle, changes, _rel_cells, _top, _left in candidates:
                self.place(board, changes, mark)
                if self._dfs(board, remaining, depth + 1):
                    return True
                self.unplace(board, changes)
            remaining[group_idx] += 1
        return False

    def _count_dfs(
        self,
        board: List[List[int]],
        remaining: List[int],
        max_solutions: Optional[int],
        progress_callback: Optional[ProgressCallback] = None,
        depth: int = 0,
    ) -> int:
        self.node_count += 1
        filled = sum(cell != 0 for row in board for cell in row)
        self.best_depth = max(self.best_depth, filled)

        target = self.first_empty(board)
        if target is None:
            solved = 1 if all(rem == 0 for rem in remaining) else 0
            if solved and self.first_solution_node is None:
                self.first_solution_node = self.node_count
                self.first_solution_depth = depth
                self.first_solution_filled = filled
            if solved and progress_callback is not None:
                progress_callback(1, self.node_count, depth, filled)
            return solved

        if not self.area_matches_remaining(board, remaining):
            return 0

        group_options = self.ordered_groups(board, remaining, target)
        if not group_options:
            return 0

        total = 0
        for group_idx, candidates in group_options:
            candidates.sort(key=lambda item: self._placement_score(board, item[1]))
            remaining[group_idx] -= 1
            mark = self.group_defs[group_idx]["id"]
            for _angle, changes, _rel_cells, _top, _left in candidates:
                self.place(board, changes, mark)
                gained = self._count_dfs(
                    board,
                    remaining,
                    max_solutions=max_solutions,
                    progress_callback=progress_callback,
                    depth=depth + 1,
                )
                total += gained
                self.unplace(board, changes)
                if max_solutions is not None and total >= max_solutions:
                    remaining[group_idx] += 1
                    return total
            remaining[group_idx] += 1
        return total


# ----------------------------
# Puzzle generation
# ----------------------------

BASE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2)],
    [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)],
]


def normalize_cells(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    cells = list(cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def rotate_cells_90(cells: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    cells = list(cells)
    max_r = max(r for r, _ in cells)
    return [(c, max_r - r) for r, c in cells]


def unique_rotations_from_cells(cells: Iterable[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], ...]]:
    out = []
    seen = set()

    for rot in (0, 180):
        current = list(cells)
        if rot == 180:
            current = rotate_cells_90(rotate_cells_90(current))

        key = normalize_cells(current)
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out


LIBRARY_ROTATIONS = [unique_rotations_from_cells(shape) for shape in BASE_SHAPES]


class PuzzleGenerator:
    def __init__(self, rows: int, cols: int, seed: Optional[int] = None):
        self.rows = rows
        self.cols = cols
        self.rng = random.Random(seed)
        self.node_count = 0
        self.solution_board: Optional[List[List[int]]] = None

    def first_empty(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] == 0:
                    return (r, c)
        return None

    def empty_components(self, board: List[List[int]]) -> List[int]:
        seen = [[False] * self.cols for _ in range(self.rows)]
        comps = []
        for r in range(self.rows):
            for c in range(self.cols):
                if board[r][c] != 0 or seen[r][c]:
                    continue
                stack = [(r, c)]
                seen[r][c] = True
                size = 0
                while stack:
                    cr, cc = stack.pop()
                    size += 1
                    for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and board[nr][nc] == 0 and not seen[nr][nc]:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
                comps.append(size)
        return comps

    def can_place(
        self,
        board: List[List[int]],
        rel_cells: Tuple[Tuple[int, int], ...],
        top: int,
        left: int,
    ) -> Optional[List[Tuple[int, int]]]:
        changes = []
        for dr, dc in rel_cells:
            r = top + dr
            c = left + dc
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return None
            if board[r][c] != 0:
                return None
            changes.append((r, c))
        return changes

    def random_connected_cells(self, size: int) -> Tuple[Tuple[int, int], ...]:
        """Generate one random connected polyomino with the requested area.

        The shape is normalized to top-left coordinates. This removes the old
        restriction that generated pieces must come from BASE_SHAPES.
        """
        if size <= 0:
            raise ValueError("size must be positive")

        cells = {(0, 0)}
        while len(cells) < size:
            r, c = self.rng.choice(list(cells))
            nr, nc = self.rng.choice(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
            cells.add((nr, nc))
        return normalize_cells(cells)

    def random_candidate_shapes(self, size_choices, n_shapes: int = 64) -> List[Tuple[Tuple[int, int], ...]]:
        """Generate a pool of random connected candidate shapes.

        Shapes are deduplicated after normalization. The pool is regenerated at
        every DFS node, so generation is not limited to the fixed library.
        """
        choices = list(size_choices)
        seen = set()
        out = []
        attempts = max(n_shapes * 8, 32)
        for _ in range(attempts):
            size = self.rng.choice(choices)
            shape = self.random_connected_cells(size)
            if shape in seen:
                continue
            seen.add(shape)
            out.append(shape)
            if len(out) >= n_shapes:
                break
        return out

    def generate(
        self,
        target_piece_count: Optional[int] = None,
        max_nodes: int = 300000,
        shape_mode: str = "free",
        shape_candidates: int = 64,
        piece_sizes: Iterable[int] = (2, 3, 4, 5),
    ) -> Optional[List[List[List[int]]]]:
        board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        placements: List[List[Tuple[int, int]]] = []
        total_area = self.rows * self.cols
        size_choices = {int(size) for size in piece_sizes}
        if not size_choices or min(size_choices) < 1:
            raise ValueError("piece_sizes must contain positive integers")
        min_piece_size = min(size_choices)
        max_piece_size = max(size_choices)

        # The old library-based generator narrowed piece sizes based on the
        # average target area. That is okay for the fixed BASE_SHAPES setting,
        # but it is too restrictive for free-shape generation. For example, a
        # 7x13 board with target_pieces=24 has area 91, so it needs a mixture
        # that includes size-3 pieces; forcing only {4, 5} makes generation
        # impossible. Therefore, keep all sizes for free mode and only apply
        # the old heuristic to library mode.
        if target_piece_count and shape_mode == "library":
            avg = total_area / target_piece_count
            # Library mode can only use sizes that both exist in the library
            # and were requested by the user.
            library_sizes = {len(shape) for rotations in LIBRARY_ROTATIONS for shape in rotations}
            size_choices &= library_sizes
            if not size_choices:
                raise ValueError("none of --piece-sizes exists in the shape library")
            min_piece_size = min(size_choices)
            max_piece_size = max(size_choices)

        def recurse() -> bool:
            self.node_count += 1
            if self.node_count > max_nodes:
                return False
            target = self.first_empty(board)
            if target is None:
                if target_piece_count is None:
                    return True
                return len(placements) == target_piece_count

            empty = sum(1 for row in board for cell in row if cell == 0)
            dynamic_size_choices = set(size_choices)
            if target_piece_count is not None:
                pieces_left = target_piece_count - len(placements)
                if pieces_left <= 0:
                    return False
                if empty < min_piece_size * pieces_left or empty > max_piece_size * pieces_left:
                    return False

                # Only try piece sizes that still allow the remaining cells to be
                # partitioned into the remaining number of pieces. This is very
                # important when generating free random shapes, because otherwise
                # the DFS wastes a lot of time trying sizes that make the final
                # piece-count target impossible.
                dynamic_size_choices = {
                    s for s in size_choices
                    if empty - s >= min_piece_size * (pieces_left - 1)
                    and empty - s <= max_piece_size * (pieces_left - 1)
                }
                if not dynamic_size_choices:
                    return False

            comps = self.empty_components(board)
            if any(comp == 1 for comp in comps):
                return False

            tr, tc = target
            candidate_moves = []

            if shape_mode == "library":
                shape_pool = []
                shape_order = list(range(len(LIBRARY_ROTATIONS)))
                self.rng.shuffle(shape_order)
                for shape_idx in shape_order:
                    rotations = LIBRARY_ROTATIONS[shape_idx][:]
                    self.rng.shuffle(rotations)
                    shape_pool.extend(rotations)
            elif shape_mode == "free":
                shape_pool = self.random_candidate_shapes(dynamic_size_choices, n_shapes=shape_candidates)
            else:
                raise ValueError(f"Unknown shape_mode: {shape_mode}")

            for rel_cells in shape_pool:
                if len(rel_cells) not in dynamic_size_choices:
                    continue
                for ar, ac in rel_cells:
                    top = tr - ar
                    left = tc - ac
                    changes = self.can_place(board, rel_cells, top, left)
                    if changes is not None:
                        candidate_moves.append(changes)
            dedup = []
            seen = set()
            for changes in candidate_moves:
                key = tuple(sorted(changes))
                if key not in seen:
                    seen.add(key)
                    dedup.append(changes)
            self.rng.shuffle(dedup)
            dedup.sort(key=len, reverse=True)

            for changes in dedup:
                mark = len(placements) + 1
                for r, c in changes:
                    board[r][c] = mark
                placements.append(changes)
                if recurse():
                    return True
                placements.pop()
                for r, c in changes:
                    board[r][c] = 0
            return False

        ok = recurse()
        if not ok:
            return None
        self.solution_board = [row[:] for row in board]
        return [cells_to_piece(changes) for changes in placements]


def write_puzzle_file(pieces: List[List[List[int]]], path: str, rows: Optional[int] = None, cols: Optional[int] = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if rows is not None:
            f.write(f"# rows {rows}\n")
        if cols is not None:
            f.write(f"# cols {cols}\n")
        if rows is not None or cols is not None:
            f.write("\n")
        for idx, piece in enumerate(pieces):
            for row in piece:
                f.write("".join(str(x) for x in row) + "\n")
            if idx != len(pieces) - 1:
                f.write("\n")


def write_solution_file(board: List[List[int]], path: str, puzzle_path: str) -> None:
    """Write the known tiling used to construct a generated puzzle.

    Each integer is the 1-based piece number in puzzle-file order. This does
    not run an extra solver: generation itself already constructs a valid
    full-board tiling.
    """
    width = max(2, len(str(max(max(row) for row in board))))
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# solution for {os.path.basename(puzzle_path)}\n")
        f.write(f"# rows {len(board)}\n")
        f.write(f"# cols {len(board[0]) if board else 0}\n")
        f.write("# cell value = piece number in puzzle-file order\n\n")
        for row in board:
            f.write(" ".join(f"{cell:{width}d}" for cell in row) + "\n")


def auto_output_name(rows: int, cols: int, target_pieces: Optional[int], output: Optional[str]) -> str:
    """Return an output filename whose numeric suffix matches rows/cols/target pieces.

    If output is None, "auto", the old default "generated_puzzle.txt", or an
    easy_nowrap*.txt filename, the filename is automatically normalized to:
        easy_nowrap{rows}{cols}{target_pieces}.txt

    Custom filenames that do not match easy_nowrap*.txt are preserved.
    """
    target_text = str(target_pieces) if target_pieces is not None else "x"
    auto_name = f"easy_nowrap{rows}{cols}{target_text}.txt"

    if output is None or output == "auto" or output == "generated_puzzle.txt":
        return auto_name

    # If the user passed something like easy_nowrap71324.txt, update the numbers
    # to match the actual --rows/--cols/--target-pieces inputs. Keep the folder.
    dirname = ""
    basename = output
    slash = max(output.rfind("/"), output.rfind("\\"))
    if slash != -1:
        dirname = output[:slash + 1]
        basename = output[slash + 1:]

    if re.fullmatch(r"easy_nowrap\d+\.txt", basename):
        return dirname + auto_name

    return output


def numeric_folder_name(rows: int, cols: int, target_pieces: Optional[int]) -> str:
    """Folder name based on the user-provided numeric puzzle settings.

    Examples:
        rows=7, cols=13, target_pieces=24 -> "71324"
        rows=4, cols=8, target_pieces=10  -> "4810"
    """
    target_text = str(target_pieces) if target_pieces is not None else "x"
    return f"{rows}{cols}{target_text}"


def output_directory(rows: int, cols: int, target_pieces: Optional[int], output: str) -> str:
    """Return the directory where generated puzzle files should be placed.

    The folder is named by the puzzle numbers, e.g. 71324 for 7x13 with
    24 target pieces. If --output includes a parent directory, create the
    numeric folder under that parent directory.
    """
    folder = numeric_folder_name(rows, cols, target_pieces)
    parent = os.path.dirname(output)
    return os.path.join(parent, folder) if parent else folder


def numbered_output_path(output: str, index: int, total: int, out_dir: Optional[str] = None) -> str:
    """Return output path for single or batch generation.

    All generated files are placed inside out_dir when provided.
    """
    filename = os.path.basename(output)
    if total <= 1:
        out_name = filename
    else:
        dot = filename.rfind(".")
        if dot == -1:
            out_name = f"{filename}_{index:03d}"
        else:
            out_name = f"{filename[:dot]}_{index:03d}{filename[dot:]}"
    return os.path.join(out_dir, out_name) if out_dir else out_name


def verify_puzzle_file(path: str, rows: int, cols: int, no_wrap: bool) -> Tuple[bool, int]:
    grouped_pieces = read_pieces(path)
    game = InteractiveCornGame(grouped_pieces, rows=rows, cols=cols, wrap_cols=not no_wrap)
    solver = game.solver()
    ok, _solved_board, nodes = solver.solve_from_state(game.board, [g["remaining"] for g in game.groups])
    return ok, nodes


# ----------------------------
# Interactive shell
# ----------------------------

class InteractiveCornGame:
    def __init__(self, grouped_pieces: List[List[List[List[int]]]], rows: int = 7, cols: int = 14, wrap_cols: bool = True):
        self.rows = rows
        self.cols = cols
        self.wrap_cols = wrap_cols
        self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.placements: List[dict] = []
        self.groups = self._prepare_groups(grouped_pieces)

    def _prepare_groups(self, grouped_pieces: List[List[List[List[int]]]]) -> List[dict]:
        groups = []
        for idx, same_list in enumerate(grouped_pieces, start=1):
            base = same_list[0]
            variants = generate_variants(base)
            groups.append({
                "id": idx,
                "count": len(same_list),
                "remaining": len(same_list),
                "area": count_area(base),
                "variants": variants,
                "base": piece_cells(base),
            })
        return groups

    def solver(self) -> ExactishSolver:
        group_defs = [{
            "id": g["id"],
            "area": g["area"],
            "variants": g["variants"],
        } for g in self.groups]
        return ExactishSolver(self.rows, self.cols, self.wrap_cols, group_defs)

    def show_board(self) -> None:
        print("    " + " ".join(f"{c:02d}" for c in range(self.cols)))
        for r, row in enumerate(self.board):
            cells = []
            for cell in row:
                if cell == 0:
                    cells.append("· ")
                else:
                    sym = symbol_for_group(cell)
                    color = COLOR_MAP.get(((cell - 1) % len(COLOR_MAP)) + 1, "")
                    cells.append(f"{color}{sym} {RESET}" if color else f"{sym} ")
            print(f"{r:02d}  " + "".join(cells))

    def show_pieces(self) -> None:
        print("Available piece groups:")
        for g in self.groups:
            print(f"  piece {g['id']}: remaining={g['remaining']}/{g['count']}, area={g['area']}, rotations={[ang for ang, _ in g['variants']]}")
            for line in render_piece_from_cells(g["base"]):
                print("    " + line)

    def can_place(self, rel_cells: Tuple[Tuple[int, int], ...], top: int, left: int) -> Optional[List[Tuple[int, int]]]:
        changes = []
        for dr, dc in rel_cells:
            r = top + dr
            c = left + dc
            if self.wrap_cols:
                c %= self.cols
            if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
                return None
            if self.board[r][c] != 0:
                return None
            changes.append((r, c))
        return changes

    def place(self, piece_id: int, angle: int, top: int, left: int) -> None:
        if not (1 <= piece_id <= len(self.groups)):
            print("Invalid piece id.")
            return
        group = self.groups[piece_id - 1]
        if group["remaining"] <= 0:
            print("No copies remaining for that piece group.")
            return
        rel_cells = None
        for ang, cells in group["variants"]:
            if ang == angle:
                rel_cells = cells
                break
        if rel_cells is None:
            print(f"Rotation {angle} not available. Use one of {[ang for ang, _ in group['variants']]}")
            return
        changes = self.can_place(rel_cells, top, left)
        if changes is None:
            print("Cannot place piece there (out of bounds or overlap).")
            return
        for r, c in changes:
            self.board[r][c] = piece_id
        group["remaining"] -= 1
        self.placements.append({
            "piece_id": piece_id,
            "angle": angle,
            "top": top,
            "left": left,
            "changes": changes,
        })
        print(f"Placed piece {piece_id} at row={top}, col={left}, rotation={angle}.")

    def undo(self) -> None:
        if not self.placements:
            print("Nothing to undo.")
            return
        move = self.placements.pop()
        for r, c in move["changes"]:
            self.board[r][c] = 0
        self.groups[move["piece_id"] - 1]["remaining"] += 1
        print(f"Removed last placement: piece {move['piece_id']}.")

    def reset(self) -> None:
        self.board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.placements.clear()
        for g in self.groups:
            g["remaining"] = g["count"]
        print("Board reset.")

    def status(self) -> None:
        filled = sum(cell != 0 for row in self.board for cell in row)
        total = self.rows * self.cols
        rem = sum(g["remaining"] for g in self.groups)
        remaining_area = sum(g["remaining"] * g["area"] for g in self.groups)
        empty = total - filled
        print(f"Filled cells: {filled}/{total} | remaining piece copies: {rem} | empty cells: {empty} | remaining piece area: {remaining_area}")
        if filled == total and rem == 0:
            print("Puzzle solved.")
        elif remaining_area != empty:
            print("Current state cannot lead to a full-board tiling: remaining piece area does not match remaining empty cells.")

    def auto_solve(self) -> None:
        remaining = [g["remaining"] for g in self.groups]
        solver = self.solver()
        ok, solved_board, nodes = solver.solve_from_state(self.board, remaining)
        if ok:
            self.board = solved_board
            for g in self.groups:
                g["remaining"] = 0
            print(
                f"Solver found a solution from the current board. "
                f"Explored {nodes} nodes. "
                f"First solution node={solver.first_solution_node}, "
                f"solution depth={solver.first_solution_depth}, "
                f"filled cells={solver.first_solution_filled}."
            )
        else:
            print(f"No solution found from the current board. Explored {nodes} nodes.")

    def count_solutions(
        self,
        max_solutions: Optional[int] = None,
        report_every: int = 1,
        report_every_nodes: int = 0,
    ) -> None:
        remaining = [g["remaining"] for g in self.groups]
        solver = self.solver()
        found = 0
        start = time.time()
        last_report_nodes = 0

        def progress(delta: int, nodes: int, depth: int, filled: int) -> None:
            nonlocal found, last_report_nodes
            found += delta
            should_print = False
            if report_every > 0 and found % report_every == 0:
                should_print = True
            if report_every_nodes > 0 and nodes - last_report_nodes >= report_every_nodes:
                should_print = True
            if should_print:
                elapsed = time.time() - start
                print(
                    f"[progress] solutions={found}, "
                    f"nodes={nodes}, "
                    f"solution_depth={depth}, "
                    f"filled={filled}, "
                    f"elapsed={elapsed:.2f}s",
                    flush=True,
                )
                last_report_nodes = nodes

        count, nodes = solver.count_solutions_from_state(
            self.board,
            remaining,
            max_solutions=max_solutions,
            progress_callback=progress,
        )
        elapsed = time.time() - start
        limit_note = "" if max_solutions is None else f" (stopped after reaching limit {max_solutions})"
        print(
            f"Total solutions from the current board: {count}{limit_note}. "
            f"Explored {nodes} nodes in {elapsed:.2f}s. "
            f"First solution node={solver.first_solution_node}, "
            f"solution depth={solver.first_solution_depth}, "
            f"filled cells={solver.first_solution_filled}."
        )

    def help(self) -> None:
        print(
            "Commands:\n"
            "  help                                 Show this help\n"
            "  board                                Print the board\n"
            "  pieces                               Show piece groups and remaining counts\n"
            "  place <id> <rot> <r> <c>             Place a piece using top-left anchor\n"
            "                                       rot can be 0/90/180/270 if available\n"
            "  undo                                 Undo the last move\n"
            "  reset                                Clear the board and restore all pieces\n"
            "  solve                                Solve from the current board state\n"
            "  count [limit] [report_every]         Count all solutions and print progress\n"
            "  countnodes <limit> <report_nodes>    Count solutions, printing every N nodes\n"
            "  status                               Show progress\n"
            "  quit                                 Exit\n"
        )

    def repl(self) -> None:
        print("Interactive Corn Puzzle")
        print(f"Board size: {self.rows} x {self.cols} | horizontal wrap: {self.wrap_cols}")
        print("Type 'help' for commands.\n")
        self.show_board()
        while True:
            try:
                raw = input("\ncommand> ").strip()
            except EOFError:
                print()
                break
            if not raw:
                continue
            parts = raw.split()
            cmd = parts[0].lower()
            try:
                if cmd == "help":
                    self.help()
                elif cmd == "board":
                    self.show_board()
                elif cmd == "pieces":
                    self.show_pieces()
                elif cmd == "place":
                    if len(parts) != 5:
                        print("Usage: place <id> <rot> <row> <col>")
                        continue
                    self.place(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                    self.show_board()
                elif cmd == "undo":
                    self.undo()
                    self.show_board()
                elif cmd == "reset":
                    self.reset()
                    self.show_board()
                elif cmd == "solve":
                    self.auto_solve()
                    self.show_board()
                elif cmd == "count":
                    limit = None
                    report_every = 1
                    if len(parts) >= 2:
                        limit = int(parts[1])
                    if len(parts) >= 3:
                        report_every = int(parts[2])
                    if len(parts) > 3:
                        print("Usage: count [limit] [report_every]")
                        continue
                    self.count_solutions(limit, report_every=report_every)
                elif cmd == "countnodes":
                    if len(parts) not in (2, 3):
                        print("Usage: countnodes <report_nodes> [limit]")
                        continue
                    report_nodes = int(parts[1])
                    limit = int(parts[2]) if len(parts) == 3 else None
                    self.count_solutions(limit, report_every=0, report_every_nodes=report_nodes)
                elif cmd == "status":
                    self.status()
                elif cmd in {"quit", "exit", "q"}:
                    break
                else:
                    print("Unknown command. Type 'help'.")
            except ValueError:
                print("Invalid numeric argument.")


# ----------------------------
# CLI
# ----------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive polyomino-style puzzle player, solver, and free-shape batch generator")
    parser.add_argument("puzzle_file", nargs="?", help="Path to piece definition file")
    parser.add_argument("--rows", type=int, default=7, help="Board rows")
    parser.add_argument("--cols", type=int, default=14, help="Board columns")
    parser.add_argument("--no-wrap", action="store_true", help="Disable horizontal wrap-around when playing/solving")
    parser.add_argument("--auto", action="store_true", help="Solve immediately without entering interactive mode")
    parser.add_argument("--count-solutions", action="store_true", help="Enumerate and count all solutions instead of stopping at the first one")
    parser.add_argument("--max-solutions", type=int, default=None, help="Optional cap when counting solutions")
    parser.add_argument("--report-every", type=int, default=1, help="When counting, print progress every N solutions found")
    parser.add_argument("--report-every-nodes", type=int, default=0, help="When counting, also print progress every N search nodes")
    parser.add_argument("--generate", action="store_true", help="Generate a new solvable puzzle instead of reading one")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for puzzle generation")
    parser.add_argument("--target-pieces", type=int, default=None, help="Exact target number of pieces when generating")
    parser.add_argument(
        "--piece-sizes",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Allowed piece areas, e.g. --piece-sizes 4 5 6",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output file for generated puzzle. If omitted, 'auto', or easy_nowrap*.txt, "
            "the filename is auto-set to easy_nowrap{rows}{cols}{target_pieces}.txt. Generated files are saved inside a folder named {rows}{cols}{target_pieces}"
        ),
    )
    parser.add_argument("--num-puzzles", type=int, default=1, help="Generate N puzzles with the same rows/cols/target settings")
    parser.add_argument("--shape-mode", choices=["free", "library"], default="free", help="Generation shape source: free=random connected shapes, library=old BASE_SHAPES")
    parser.add_argument("--shape-candidates", type=int, default=64, help="Number of random candidate shapes sampled at each DFS node when --shape-mode free")
    parser.add_argument("--max-generate-nodes", type=int, default=300000, help="Max DFS nodes allowed per generated puzzle")
    parser.add_argument("--skip-verify", action="store_true", help="Skip post-generation solver verification")
    parser.add_argument(
        "--solution-dir",
        type=str,
        default=None,
        help="Also write each generated tiling answer into this separate directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate:
        if args.num_puzzles < 1:
            print("Error: --num-puzzles must be >= 1")
            sys.exit(2)
        if not args.piece_sizes or any(size < 1 for size in args.piece_sizes):
            print("Error: --piece-sizes must contain positive integers")
            sys.exit(2)
        if args.target_pieces is not None:
            area = args.rows * args.cols
            min_area = args.target_pieces * min(args.piece_sizes)
            max_area = args.target_pieces * max(args.piece_sizes)
            if not min_area <= area <= max_area:
                print(
                    f"Error: board area {area} cannot be split into {args.target_pieces} "
                    f"pieces of sizes {sorted(set(args.piece_sizes))} "
                    f"(possible area range {min_area}..{max_area})"
                )
                sys.exit(2)

        base_seed = args.seed if args.seed is not None else int(time.time())
        made = 0
        attempts = 0
        max_attempts = max(args.num_puzzles * 20, args.num_puzzles + 10)
        resolved_output = auto_output_name(args.rows, args.cols, args.target_pieces, args.output)
        out_dir = output_directory(args.rows, args.cols, args.target_pieces, resolved_output)
        os.makedirs(out_dir, exist_ok=True)
        if args.solution_dir:
            os.makedirs(args.solution_dir, exist_ok=True)
        print(f"Output folder: {out_dir}")
        print(f"Output basename: {os.path.basename(resolved_output)}")

        while made < args.num_puzzles and attempts < max_attempts:
            seed = base_seed + attempts
            attempts += 1
            gen = PuzzleGenerator(rows=args.rows, cols=args.cols, seed=seed)
            pieces = gen.generate(
                target_piece_count=args.target_pieces,
                max_nodes=args.max_generate_nodes,
                shape_mode=args.shape_mode,
                shape_candidates=args.shape_candidates,
                piece_sizes=args.piece_sizes,
            )
            if pieces is None:
                print(f"[skip] seed={seed}: failed to generate within node limit")
                continue

            made += 1
            out_path = numbered_output_path(resolved_output, made, args.num_puzzles, out_dir=out_dir)
            write_puzzle_file(pieces, out_path, rows=args.rows, cols=args.cols)
            print(
                f"[{made}/{args.num_puzzles}] Generated puzzle with {len(pieces)} pieces "
                f"using seed={seed}, shape_mode={args.shape_mode}; wrote: {out_path}"
            )
            print(f"    Generator explored {gen.node_count} nodes.")

            if args.solution_dir:
                stem, _ext = os.path.splitext(os.path.basename(out_path))
                solution_path = os.path.join(args.solution_dir, f"{stem}_solution.txt")
                write_solution_file(gen.solution_board, solution_path, out_path)
                print(f"    Wrote construction solution: {solution_path}")

            if not args.skip_verify:
                ok, nodes = verify_puzzle_file(out_path, rows=args.rows, cols=args.cols, no_wrap=args.no_wrap)
                print(f"    Verification with current solver: {'OK' if ok else 'FAILED'} (nodes={nodes})")

        if made < args.num_puzzles:
            print(f"Only generated {made}/{args.num_puzzles} puzzles after {attempts} attempts.")
            sys.exit(1)
        return

    if not args.puzzle_file:
        print("Error: puzzle_file is required unless --generate is used.")
        sys.exit(2)

    grouped_pieces = read_pieces(args.puzzle_file)
    game = InteractiveCornGame(grouped_pieces, rows=args.rows, cols=args.cols, wrap_cols=not args.no_wrap)
    if args.count_solutions:
        game.count_solutions(
            max_solutions=args.max_solutions,
            report_every=args.report_every,
            report_every_nodes=args.report_every_nodes,
        )
    elif args.auto:
        game.auto_solve()
        game.show_board()
    else:
        game.repl()


if __name__ == "__main__":
    main()
