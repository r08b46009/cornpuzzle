#!/usr/bin/env python3
"""CornPuzzle GUI viewer / placer.

A small standalone tool (Tkinter only, no extra pip installs needed) that:
  * reads a CornPuzzle puzzle file, any filename/extension (``# rows N`` /
    ``# cols N`` header followed by blank-line separated 0/1 piece shapes),
  * draws the empty board according to the header size,
  * lists every piece at the bottom of the window in its own color,
  * lets you drag pieces with the mouse onto the board.

Run with:
    python cornpuzzle_gui.py [folder]

``folder`` should contain the puzzle files (e.g. ``cornpuzzle/71424``); any
file in it can be loaded regardless of extension, as long as its content
matches the ``# rows`` / ``# cols`` + 0/1 grid format. Use "開檔案..." in the
toolbar to load a single file from anywhere, extension-agnostic too.
If no folder is given, the script looks for a ``71424`` folder next to
itself, and otherwise lets you pick a folder from a dialog.

Controls:
  * Left-click + drag a piece from the tray onto the board to place it.
  * Left-click + drag a piece that is already on the board to move it
    (drop it back over the tray area to return it).
  * Right-click a tray piece to rotate it 90 degrees before placing.
  * Right-click a placed piece to send it back to the tray.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


def _enable_windows_dpi_awareness() -> None:
    """Tell Windows this app handles its own DPI scaling.

    Without this, Tkinter windows get bitmap-stretched by Windows on any
    scaled display (125%/150%/200%), which is what makes the UI look
    blurry. Must be called before the first Tk() is created. No-op on
    non-Windows platforms.
    """
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Pure logic (no tkinter dependency -- easy to test on its own)
# ---------------------------------------------------------------------------

Cell = tuple[int, int]


def normalize_cells(cells: list[Cell]) -> list[Cell]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def grid_lines_to_cells(grid_lines: list[str]) -> list[Cell]:
    cells = [(r, c) for r, line in enumerate(grid_lines) for c, ch in enumerate(line) if ch == '1']
    if not cells:
        raise ValueError("empty piece shape block")
    return normalize_cells(cells)


def parse_puzzle_file(path: Path) -> tuple[int, int, list[list[Cell]]]:
    """Parse a CornPuzzle txt file into (rows, cols, pieces)."""
    rows = cols = None
    pieces: list[list[Cell]] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append(grid_lines_to_cells(current))
            current.clear()

    with open(path, encoding='utf-8') as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped.startswith('#'):
                parts = stripped.lstrip('#').split()
                if len(parts) >= 2:
                    if parts[0] == 'rows':
                        rows = int(parts[1])
                    elif parts[0] == 'cols':
                        cols = int(parts[1])
                continue
            if stripped == '':
                flush()
                continue
            current.append(stripped)
    flush()

    if rows is None or cols is None:
        raise ValueError(f"{path.name}: missing '# rows' / '# cols' header")
    return rows, cols, pieces


def rotate_cells_cw(cells: list[Cell]) -> list[Cell]:
    """Rotate a normalized cell list 90 degrees clockwise."""
    max_r = max(r for r, _ in cells)
    return normalize_cells([(c, max_r - r) for r, c in cells])


def cells_bbox(cells: list[Cell]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def wrap_col(c: int, cols: int) -> int:
    """Wrap a column index into [0, cols); Python's % already handles negatives."""
    return c % cols


def resolve_board_cells(cells: list[Cell], origin: Cell, rows: int, cols: int) -> list[Cell] | None:
    """Map piece-local cells to board cells, wrapping columns left/right
    (matching the C++ env's wrapColActive). Rows never wrap -- if any cell's
    row falls outside the board, the whole placement is invalid (None)."""
    orow, ocol = origin
    resolved = []
    for dr, dc in cells:
        r = orow + dr
        if not (0 <= r < rows):
            return None
        resolved.append((r, wrap_col(ocol + dc, cols)))
    return resolved


def fits_on_board(board_cells: list[Cell] | None, occupied: set[Cell]) -> bool:
    if board_cells is None:
        return False
    return len(set(board_cells)) == len(board_cells) and all(cell not in occupied for cell in board_cells)


def make_palette(n: int) -> list[str]:
    """Generate `n` visually distinct hex colors around the HSV wheel."""
    colors = []
    n = max(n, 1)
    for i in range(n):
        h = (i * 0.61803398875) % 1.0  # golden-ratio spacing looks less repetitive than i/n
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.90)
        colors.append('#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)))
    return colors


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

BOARD_CELL = 34
TRAY_CELL = 13
SLOT_SIZE = 96
MARGIN = 20
TRAY_COLS = 9  # how many piece slots per tray row


class Piece:
    def __init__(self, piece_id: int, base_cells: list[Cell], color: str):
        self.id = piece_id
        self.color = color
        self.rotations = [base_cells]
        cells = base_cells
        for _ in range(3):
            cells = rotate_cells_cw(cells)
            self.rotations.append(cells)
        self.rot_index = 0
        self.placed = False
        self.origin: Cell | None = None

    @property
    def cells(self) -> list[Cell]:
        return self.rotations[self.rot_index]

    def rotate(self) -> None:
        self.rot_index = (self.rot_index + 1) % 4


def find_default_folder(script_path: Path) -> Path | None:
    candidates = [
        script_path.parent / '71424',
        script_path.parent.parent / 'cornpuzzle' / '71424',
        script_path.parent.parent.parent / 'cornpuzzle' / '71424',
        Path.cwd() / '71424',
        Path.cwd() / 'cornpuzzle' / '71424',
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.is_file() for p in candidate.iterdir()):
            return candidate
    return None


class CornPuzzleApp:
    def __init__(self, root: tk.Tk, folder: Path | None):
        self.root = root
        self.root.title("CornPuzzle Viewer")
        self.folder: Path | None = None
        self.rows = 0
        self.cols = 0
        self.pieces: list[Piece] = []
        self.board: dict[Cell, int] = {}  # (r, c) -> piece_id

        self.drag_piece: Piece | None = None
        self.drag_from: str | None = None  # 'tray' or 'board'
        self.drag_last_target: tuple[Cell, bool] | None = None  # (origin, legal)

        self._build_widgets()
        if folder is not None:
            self.load_folder(folder)
        else:
            found = find_default_folder(Path(__file__).resolve())
            if found is not None:
                self.load_folder(found)
            else:
                self.root.after(200, self.browse_folder)

    # -- widget setup -----------------------------------------------------
    def _build_widgets(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(side='top', fill='x')

        ttk.Button(top, text="開資料夾...", command=self.browse_folder).pack(side='left')
        ttk.Button(top, text="開檔案...", command=self.browse_file).pack(side='left', padx=(4, 0))

        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(top, textvariable=self.file_var, state='readonly', width=32)
        self.file_combo.pack(side='left', padx=6)
        self.file_combo.bind('<<ComboboxSelected>>', lambda _e: self.load_file(self.file_var.get()))

        ttk.Button(top, text="重置 Reset", command=self.reset_current_file).pack(side='left', padx=6)

        self.status_var = tk.StringVar(value="尚未載入拼圖")
        ttk.Label(top, textvariable=self.status_var).pack(side='left', padx=12)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side='top', fill='both', expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg='white', width=980, height=700, highlightthickness=0)
        vbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        vbar.pack(side='right', fill='y')

        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_motion)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<ButtonPress-3>', self.on_right_click)

        self.board_x0 = MARGIN
        self.board_y0 = MARGIN
        self.tray_y0 = MARGIN

    # -- file / folder handling --------------------------------------------
    def browse_folder(self) -> None:
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="選擇存放拼圖檔案的資料夾")
        if folder:
            self.load_folder(Path(folder))

    def browse_file(self) -> None:
        from tkinter import filedialog
        initial_dir = str(self.folder) if self.folder is not None else None
        chosen = filedialog.askopenfilename(
            title="選擇拼圖檔案(不限副檔名)",
            initialdir=initial_dir,
            filetypes=[("All files", "*.*")])
        if not chosen:
            return
        path = Path(chosen)
        self.load_folder(path.parent, select=path.name)

    def load_folder(self, folder: Path, select: str | None = None) -> None:
        files = sorted(p.name for p in folder.iterdir() if p.is_file())
        if not files:
            messagebox.showerror("找不到檔案", f"{folder} 裡沒有檔案")
            return
        self.folder = folder
        self.file_combo['values'] = files
        target = select if select in files else files[0]
        self.file_var.set(target)
        self.load_file(target)

    def reset_current_file(self) -> None:
        if self.file_var.get():
            self.load_file(self.file_var.get())

    def load_file(self, filename: str) -> None:
        assert self.folder is not None
        path = self.folder / filename
        try:
            rows, cols, piece_cells_list = parse_puzzle_file(path)
        except Exception as exc:  # noqa: BLE001 - show any parse error to the user
            messagebox.showerror("讀取失敗", f"{path.name}: {exc}")
            return

        self.rows, self.cols = rows, cols
        colors = make_palette(len(piece_cells_list))
        self.pieces = [Piece(i, cells, colors[i]) for i, cells in enumerate(piece_cells_list)]
        self.board = {}
        self.drag_piece = None
        self.drag_from = None
        self.redraw()

    # -- geometry helpers ---------------------------------------------------
    def occupied_cells(self, exclude_piece_id: int | None = None) -> set[Cell]:
        return {cell for cell, pid in self.board.items() if pid != exclude_piece_id}

    def board_cell_at(self, x: float, y: float) -> Cell | None:
        col = int((x - self.board_x0) // BOARD_CELL)
        row = int((y - self.board_y0) // BOARD_CELL)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def tray_slot_at(self, x: float, y: float) -> int | None:
        if y < self.tray_y0:
            return None
        col = int((x - MARGIN) // SLOT_SIZE)
        row = int((y - self.tray_y0) // SLOT_SIZE)
        if col < 0 or col >= TRAY_COLS or row < 0:
            return None
        return row * TRAY_COLS + col

    # -- drawing --------------------------------------------------------
    def redraw(self, mouse_xy: tuple[float, float] | None = None) -> None:
        self.canvas.delete('all')

        # board
        for r in range(self.rows):
            for c in range(self.cols):
                x1 = self.board_x0 + c * BOARD_CELL
                y1 = self.board_y0 + r * BOARD_CELL
                pid = self.board.get((r, c))
                fill = self.pieces[pid].color if pid is not None else '#f2f2f2'
                self.canvas.create_rectangle(x1, y1, x1 + BOARD_CELL, y1 + BOARD_CELL, fill=fill, outline='#aaaaaa')

        board_bottom = self.board_y0 + self.rows * BOARD_CELL
        self.tray_y0 = board_bottom + MARGIN + 20

        self.canvas.create_text(
            MARGIN, board_bottom + MARGIN, anchor='w', font=('TkDefaultFont', 11, 'bold'),
            text="拼圖片(拖曳到棋盤上放置;右鍵:未放置的旋轉 / 已放置的收回)")

        unplaced = [p for p in self.pieces if not p.placed and p is not self.drag_piece]
        self.tray_slot_pieces: dict[int, Piece] = {}
        for idx, piece in enumerate(unplaced):
            self.tray_slot_pieces[idx] = piece
            row, col = divmod(idx, TRAY_COLS)
            sx = MARGIN + col * SLOT_SIZE
            sy = self.tray_y0 + row * SLOT_SIZE
            self.canvas.create_rectangle(sx, sy, sx + SLOT_SIZE - 8, sy + SLOT_SIZE - 8, outline='#cccccc')
            self._draw_piece_cells(piece.cells, sx, sy, SLOT_SIZE - 8, SLOT_SIZE - 8, TRAY_CELL, piece.color)

        # drag ghost, drawn last (on top)
        if self.drag_piece is not None and mouse_xy is not None:
            self._draw_drag_ghost(mouse_xy)

        total = len(self.pieces)
        placed = sum(1 for p in self.pieces if p.placed)
        state = "已完成! 所有拼圖片都放好了" if placed == total and total > 0 and self._board_full() else \
            f"{self.rows} x {self.cols} 盤面 | 已放置 {placed}/{total} 片"
        self.status_var.set(state)

        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _board_full(self) -> bool:
        return len(self.board) == self.rows * self.cols

    def _draw_piece_cells(self, cells: list[Cell], sx: float, sy: float, slot_w: float, slot_h: float,
                          cell_size: float, color: str, outline: str = 'black') -> None:
        h, w = cells_bbox(cells)
        offset_x = sx + max(0, (slot_w - w * cell_size)) / 2
        offset_y = sy + max(0, (slot_h - h * cell_size)) / 2
        for dr, dc in cells:
            px = offset_x + dc * cell_size
            py = offset_y + dr * cell_size
            self.canvas.create_rectangle(px, py, px + cell_size, py + cell_size, fill=color, outline=outline)

    def _draw_ghost_cell(self, r: int, c: int, piece: Piece, outline: str) -> None:
        x1 = self.board_x0 + c * BOARD_CELL
        y1 = self.board_y0 + r * BOARD_CELL
        self.canvas.create_rectangle(
            x1 + 2, y1 + 2, x1 + BOARD_CELL - 2, y1 + BOARD_CELL - 2,
            fill=piece.color, outline=outline, width=2, stipple='gray50')

    def _draw_drag_ghost(self, mouse_xy: tuple[float, float]) -> None:
        piece = self.drag_piece
        assert piece is not None
        x, y = mouse_xy
        cell = self.board_cell_at(x, y)
        if cell is not None:
            # columns wrap left/right (matches the C++ env's wrapColActive);
            # rows never wrap, so resolve_board_cells returns None if the
            # piece would stick out past the top/bottom edge.
            board_cells = resolve_board_cells(piece.cells, cell, self.rows, self.cols)
            if board_cells is not None:
                legal = fits_on_board(board_cells, self.occupied_cells())
                self.drag_last_target = (cell, legal)
                outline = '#00aa00' if legal else '#cc0000'
                for r, c in board_cells:
                    self._draw_ghost_cell(r, c, piece, outline)
            else:
                self.drag_last_target = None
                for dr, dc in piece.cells:
                    r = cell[0] + dr
                    if 0 <= r < self.rows:
                        self._draw_ghost_cell(r, wrap_col(cell[1] + dc, self.cols), piece, '#cc0000')
        else:
            self.drag_last_target = None
            h, w = cells_bbox(piece.cells)
            self._draw_piece_cells(piece.cells, x - w * TRAY_CELL / 2, y - h * TRAY_CELL / 2,
                                   w * TRAY_CELL, h * TRAY_CELL, TRAY_CELL, piece.color, outline='#666666')

    # -- mouse handling ---------------------------------------------------
    def on_press(self, event: tk.Event) -> None:
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        cell = self.board_cell_at(x, y)
        if cell is not None and cell in self.board:
            pid = self.board[cell]
            piece = self.pieces[pid]
            self.board = {rc: p for rc, p in self.board.items() if p != pid}
            piece.placed = False
            piece.origin = None
            self.drag_piece = piece
            self.drag_from = 'board'
            self.redraw((x, y))
            return

        slot = self.tray_slot_at(x, y)
        if slot is not None and slot in getattr(self, 'tray_slot_pieces', {}):
            piece = self.tray_slot_pieces[slot]
            self.drag_piece = piece
            self.drag_from = 'tray'
            self.redraw((x, y))

    def on_motion(self, event: tk.Event) -> None:
        if self.drag_piece is None:
            return
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.redraw((x, y))

    def on_release(self, event: tk.Event) -> None:
        if self.drag_piece is None:
            return
        piece = self.drag_piece
        target = self.drag_last_target
        if target is not None:
            origin, legal = target
            if legal:
                board_cells = resolve_board_cells(piece.cells, origin, self.rows, self.cols)
                assert board_cells is not None  # legal was computed from this same resolution
                for rc in board_cells:
                    self.board[rc] = piece.id
                piece.placed = True
                piece.origin = origin
        self.drag_piece = None
        self.drag_from = None
        self.drag_last_target = None
        self.redraw()

    def on_right_click(self, event: tk.Event) -> None:
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        cell = self.board_cell_at(x, y)
        if cell is not None and cell in self.board:
            pid = self.board[cell]
            piece = self.pieces[pid]
            self.board = {rc: p for rc, p in self.board.items() if p != pid}
            piece.placed = False
            piece.origin = None
            self.redraw()
            return

        slot = self.tray_slot_at(x, y)
        if slot is not None and slot in getattr(self, 'tray_slot_pieces', {}):
            self.tray_slot_pieces[slot].rotate()
            self.redraw()


def main() -> None:
    _enable_windows_dpi_awareness()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', nargs='?', default=None, help="folder containing puzzle files (any extension)")
    args = parser.parse_args()

    root = tk.Tk()
    folder = Path(args.folder) if args.folder else None
    CornPuzzleApp(root, folder)
    root.mainloop()


if __name__ == '__main__':
    main()
