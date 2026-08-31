#!/usr/bin/env python3
import argparse
import csv
import html
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
HEADER_RE = re.compile(r"^\s+(?:\d+\s+)+$")
ROW_RE = re.compile(r"^\s*(\d+)\s+(.*)$")
STEP_RE = re.compile(r"^step\s+(\d+):\s*(.*)$")
EVAL_RE = re.compile(r"^\[EvalScore\]\s+(.+)$")
REMAINING_RE = re.compile(r"P\d+=(\d+)")

PIECE_COLORS = [
    "#e53935", "#fb8c00", "#fdd835", "#43a047", "#00acc1", "#1e88e5",
    "#8e24aa", "#d81b60", "#7cb342", "#039be5", "#f4511e", "#5e35b1",
    "#00897b", "#c0ca33", "#6d4c41", "#546e7a",
]

def strip_ansi(text):
    return ANSI_RE.sub("", text)


def piece_color(token):
    if token == ".":
        return "#f8fafc"
    if token == "#":
        return "#334155"
    try:
        return PIECE_COLORS[(int(token) - 1) % len(PIECE_COLORS)]
    except ValueError:
        return "#cbd5e1"


def parse_board(lines, start):
    i = start
    while i < len(lines) and not HEADER_RE.match(strip_ansi(lines[i])):
        i += 1
    if i >= len(lines):
        return None, start

    cols = [int(x) for x in strip_ansi(lines[i]).split()]
    board = []
    i += 1
    while i < len(lines):
        plain = strip_ansi(lines[i])
        match = ROW_RE.match(plain)
        if not match:
            break
        cells = match.group(2).split()
        if len(cells) != len(cols):
            break
        board.append({"row": int(match.group(1)), "cells": cells})
        i += 1

    if not board:
        return None, start
    return {"cols": cols, "rows": board}, i


def parse_frames(text):
    lines = text.splitlines()
    eval_score = "N/A"
    for line in lines:
        match = EVAL_RE.match(strip_ansi(line))
        if match:
            eval_score = match.group(1).strip()

    puzzle = ""
    frames = []
    i = 0
    while i < len(lines):
        plain = strip_ansi(lines[i])
        if plain.startswith("[Puzzle]"):
            puzzle = plain[len("[Puzzle]"):].strip()
            i += 1
            continue

        title = None
        if plain == "[Initial]":
            title = "Initial"
        else:
            step_match = STEP_RE.match(plain)
            if step_match:
                title = f"Step {step_match.group(1)}: {step_match.group(2)}"

        if title is None:
            i += 1
            continue

        i += 1
        meta = []
        while i < len(lines) and not HEADER_RE.match(strip_ansi(lines[i])):
            plain_meta = strip_ansi(lines[i])
            if plain_meta:
                meta.append(plain_meta)
            i += 1

        board, i = parse_board(lines, i)
        remaining = ""
        if i < len(lines) and strip_ansi(lines[i]).startswith("remaining:"):
            remaining = strip_ansi(lines[i])
            i += 1

        if board:
            frames.append({
                "title": title,
                "meta": meta,
                "board": board,
                "remaining": remaining,
                "eval_score": eval_score,
            })

    return puzzle, eval_score, frames


def parse_remaining_total(remaining):
    return sum(int(value) for value in REMAINING_RE.findall(remaining))


def summarize_file(txt_path, eval_score, frames):
    initial_total = 0
    final_total = 0
    if frames:
        initial_total = parse_remaining_total(frames[0]["remaining"])
        final_total = parse_remaining_total(frames[-1]["remaining"])

    placed_count = max(0, initial_total - final_total)
    completion_rate = placed_count / initial_total if initial_total else 0.0
    try:
        score_value = float(eval_score)
    except ValueError:
        score_value = 0.0

    return {
        "puzzle": txt_path.stem,
        "score": score_value,
        "placed_count": placed_count,
        "total_count": initial_total,
        "placed_over_total": f"{placed_count}/{initial_total}" if initial_total else "0/0",
        "completion_rate": completion_rate,
    }


def write_summary_csv(rows, csv_path):
    if not rows:
        return

    average_score = sum(row["score"] for row in rows) / len(rows)
    average_placed = sum(row["placed_count"] for row in rows) / len(rows)
    average_total = sum(row["total_count"] for row in rows) / len(rows)
    average_completion = sum(row["completion_rate"] for row in rows) / len(rows)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "puzzle",
                "score",
                "placed_over_total",
                "completion_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "puzzle": row["puzzle"],
                "score": f"{row['score']:.6f}",
                "placed_over_total": row["placed_over_total"],
                "completion_rate": f"{row['completion_rate']:.6f}",
            })
        writer.writerow({
            "puzzle": "AVERAGE",
            "score": f"{average_score:.6f}",
            "placed_over_total": f"{average_placed:.2f}/{average_total:.2f}",
            "completion_rate": f"{average_completion:.6f}",
        })


def render_board(board):
    cols = "".join(f"<th>{col}</th>" for col in board["cols"])
    rows = []
    for row in board["rows"]:
        cells = []
        for token in row["cells"]:
            label = "" if token == "." else html.escape(token)
            classes = "cell empty" if token == "." else "cell piece"
            cells.append(
                f'<td class="{classes}" style="--piece-color:{piece_color(token)}">{label}</td>'
            )
        rows.append(f"<tr><th>{row['row']}</th>{''.join(cells)}</tr>")
    return f"<table class=\"board\"><thead><tr><th></th>{cols}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_html(source_path, puzzle, eval_score, frames):
    frame_html = []
    for frame in frames:
        meta = "".join(f"<div>{html.escape(line)}</div>" for line in frame["meta"])
        remaining = html.escape(frame["remaining"])
        frame_html.append(
            f"""
            <section class="frame">
              <div class="frame-head">
                <h2>{html.escape(frame["title"])}</h2>
                <span class="score">[EvalScore] {html.escape(frame["eval_score"])}</span>
              </div>
              <div class="meta">{meta}</div>
              {render_board(frame["board"])}
              <pre class="remaining">{remaining}</pre>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(source_path.stem)} CornPuzzle Solve</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --text: #111827;
      --muted: #64748b;
      --line: #d7dde6;
      --panel: #ffffff;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 26px;
      font-weight: 700;
    }}
    .summary {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 14px;
    }}
    .frame {{
      margin: 18px 0;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow-x: auto;
    }}
    .frame-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 8px;
    }}
    h2 {{
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }}
    .score {{
      white-space: nowrap;
      color: #0f766e;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 14px;
      font-weight: 700;
    }}
    .meta {{
      margin-bottom: 12px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      line-height: 1.45;
    }}
    .board {{
      border-collapse: separate;
      border-spacing: 3px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      table-layout: fixed;
    }}
    .board th {{
      min-width: 34px;
      height: 28px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-align: center;
    }}
    .cell {{
      width: 34px;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 5px;
      text-align: center;
      vertical-align: middle;
      font-size: 14px;
      font-weight: 800;
      box-sizing: border-box;
    }}
    .empty {{
      background: #f8fafc;
      color: transparent;
    }}
    .piece {{
      background: var(--piece-color);
      color: #111827;
      border-color: color-mix(in srgb, var(--piece-color), #000 18%);
      text-shadow: 0 1px rgba(255, 255, 255, 0.35);
    }}
    .remaining {{
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(source_path.name)}</h1>
    <p class="summary">Puzzle: {html.escape(puzzle or "unknown")} | [EvalScore] {html.escape(eval_score)} | Frames: {len(frames)}</p>
    {''.join(frame_html)}
  </main>
</body>
</html>
"""


def load_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    width, height = text_size(draw, text, font)
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2 - 1), text, font=font, fill=fill)


def render_final_png(source_path, puzzle, eval_score, frames, output_path):
    if not frames:
        return False

    final_frame = frames[-1]
    board = final_frame["board"]
    cell_size = 46
    gap = 5
    label_width = 42
    header_height = 92
    margin = 28
    col_count = len(board["cols"])
    row_count = len(board["rows"])
    board_width = label_width + gap + col_count * (cell_size + gap)
    board_height = 30 + gap + row_count * (cell_size + gap)
    width = max(760, board_width + margin * 2)
    height = header_height + board_height + margin

    image = Image.new("RGB", (width, height), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30, bold=True)
    score_font = load_font(20, bold=True)
    label_font = load_font(15)
    cell_font = load_font(18, bold=True)

    title = source_path.stem
    draw.text((margin, 22), title, font=title_font, fill="#111827")
    draw.text((margin, 58), f"[EvalScore] {eval_score}", font=score_font, fill="#0f766e")

    x0 = margin
    y0 = header_height
    draw.rounded_rectangle(
        (x0, y0, x0 + board_width, y0 + board_height),
        radius=8,
        fill="#d7dde6",
    )

    for col_idx, col in enumerate(board["cols"]):
        x = x0 + label_width + gap + col_idx * (cell_size + gap)
        draw_centered_text(draw, (x, y0 + gap, x + cell_size, y0 + 30), str(col), label_font, "#64748b")

    for row_idx, row in enumerate(board["rows"]):
        y = y0 + 30 + gap + row_idx * (cell_size + gap)
        draw_centered_text(draw, (x0 + gap, y, x0 + label_width, y + cell_size), str(row["row"]), label_font, "#64748b")
        for col_idx, token in enumerate(row["cells"]):
            x = x0 + label_width + gap + col_idx * (cell_size + gap)
            fill = piece_color(token)
            outline = "#cbd5e1" if token == "." else "#475569"
            draw.rounded_rectangle(
                (x, y, x + cell_size, y + cell_size),
                radius=7,
                fill=fill,
                outline=outline,
                width=1,
            )
            if token != ".":
                draw_centered_text(draw, (x, y, x + cell_size, y + cell_size), token, cell_font, "#111827")

    image.save(output_path)
    return True


def convert_file(txt_path, output_dir):
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    puzzle, eval_score, frames = parse_frames(text)
    html_text = render_html(txt_path, puzzle, eval_score, frames)
    html_path = output_dir / f"{txt_path.stem}.html"
    png_path = output_dir / f"{txt_path.stem}.png"
    html_path.write_text(html_text, encoding="utf-8")
    render_final_png(txt_path, puzzle, eval_score, frames, png_path)
    return html_path, png_path, len(frames), eval_score, summarize_file(txt_path, eval_score, frames)


def main():
    parser = argparse.ArgumentParser(
        description="Convert solve_cornpuzzle stdout txt files into step-by-step HTML boards."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="solve_outputs/run_001/stdout",
        help="Directory containing solve_cornpuzzle stdout .txt files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for generated HTML files. Defaults to ../html next to the stdout directory.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "html"
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = input_dir.parent / "summary.csv"
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"no .txt files found in: {input_dir}")

    summary_rows = []
    for txt_path in txt_files:
        html_path, png_path, frame_count, eval_score, summary = convert_file(txt_path, output_dir)
        summary_rows.append(summary)
        print(
            f"{txt_path} -> {html_path}, {png_path} "
            f"({frame_count} frames, [EvalScore] {eval_score})"
        )
    write_summary_csv(summary_rows, summary_csv_path)
    print(f"summary csv -> {summary_csv_path}")


if __name__ == "__main__":
    main()
